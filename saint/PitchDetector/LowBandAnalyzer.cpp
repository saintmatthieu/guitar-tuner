#include "LowBandAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "PitchDetectorLoggerInterface.h"
#include "PitchDetectorUtils.h"
#include "Utils.h"

namespace saint {
namespace {
// Top of the band this analysis has any interest in: the comb test weighs subHarmonicCombSize
// harmonics of a fundamental that can sit as high as the range floor. Everything else here is
// derived from it, which is what makes the analysis follow the tuning - the range floor moves
// with it, and so must the band, the anti-alias filter and the decimation.
float analysisTopFreq(float minFreq) {
    return subHarmonicCombSize * minFreq;
}

float antiAliasCutoff(float minFreq) {
    // A Butterworth is already 3 dB down at its own cutoff, so the anti-alias cutoff goes above the
    // band it has to pass unaltered, not on it.
    constexpr float antiAliasCutoffOverBand = 1.2f;
    return antiAliasCutoffOverBand * analysisTopFreq(minFreq);
}

// Decimate as far as those two requirements allow. Sizing it this way round - band first,
// decimation last - is what keeps them in step when the tuning changes; choosing the factor
// first leaves the cutoff to be justified after the fact, and a tuning whose comb reaches
// higher then runs into it.
int analysisDecimation(int sampleRate, float minFreq) {
    // How far above the cutoff the decimated Nyquist has to sit for what folds back to be
    // negligible: an order-6 Butterworth rolls off 36 dB per octave, so 2.6x (~1.4 octaves) puts
    // the fold-back some 50 dB down.
    constexpr float antiAliasNyquistOverCutoff = 2.6f;
    const auto nyquistNeeded = antiAliasNyquistOverCutoff * antiAliasCutoff(minFreq);
    return std::max(1, static_cast<int>(sampleRate / (2 * nyquistNeeded)));
}

// Added to the median level before it is treated as the noise floor, so that ordinary
// floor wiggle stays at or below 0 and only real partials come out positive.
constexpr float floorMarginDb = 3.f;

// Largest numerator of the rational ratios searched (see below()). Three covers the ratio the
// corpus actually shows - an estimate two thirds of the fundamental, from locking onto half of
// a dominant 3rd harmonic - without opening the field to ratios no string produces.
constexpr int maxRatioNumerator = 3;

bool isEstimateHarmonic(int m, int p) {
    return m % p == 0;
}
}  // namespace

void LowBandAnalyzer::Diagnostics::clear() {
    candidateHz.clear();
    candidateSupport.clear();
    candidateProminence.clear();
    combHz.clear();
    combProminence.clear();
    combExplained.clear();
}

LowBandAnalyzer::LowBandAnalyzer(int sampleRate, ChannelFormat channelFormat,
                                 int samplesPerBlockPerChannel, float minFreq,
                                 PitchDetectorLoggerInterface& logger, LowBandConfig config)
    : _decimationFactor(analysisDecimation(sampleRate, minFreq)),
      _channelFormat(channelFormat),
      _rate(sampleRate / _decimationFactor),
      _logger(logger),
      _lowpass(
          1, 0,
          butterworthCoefs<filterOrder>(FilterType::Lowpass, antiAliasCutoff(minFreq), sampleRate)),
      // The window is sized as if the range floor were this much lower, which is what makes it
      // long enough to resolve the harmonic comb of a fundamental an octave below the range.
      _transformer(_rate, ChannelFormat::Mono,
                   std::max(1, samplesPerBlockPerChannel / _decimationFactor),
                   minFreq / config.analysisMinFreqRatio, _transformerLogger),
      _binFreq(static_cast<float>(_rate) / _transformer.fftSize()),
      _minFrequency(minFreq / config.analysisMinFreqRatio),
      _rangeFloor(minFreq),
      _maxHarmonic(config.maxHarmonic),
      _floorBandEnd(std::min(_transformer.fftSize() / 2,
                             static_cast<int>(analysisTopFreq(minFreq) / _binFreq))) {
    _decimated.reserve(static_cast<size_t>(samplesPerBlockPerChannel) / _decimationFactor + 2);
    _spectrum.resize(static_cast<size_t>(_transformer.fftSize()));
    _floorScratch.reserve(static_cast<size_t>(_floorBandEnd));
}

void LowBandAnalyzer::process(const std::vector<float>& block) {
    // Down-mix, anti-alias filter, then keep every Dth frame. The decimation phase carries
    // across blocks, so a block count that is not a multiple of D is fine - the frame count
    // per call then varies by one, which the transformer tolerates.
    const auto numChannels = saint::numChannels(_channelFormat);
    const auto numFrames = static_cast<int>(block.size()) / numChannels;
    _decimated.clear();
    for (auto i = 0; i < numFrames; ++i) {
        auto frame = block[i * numChannels];
        for (auto c = 1; c < numChannels; ++c) {
            frame += block[i * numChannels + c];
        }
        const auto filtered = _lowpass.process(frame / numChannels);
        if (_decimationPhase == 0) {
            _decimated.push_back(filtered);
        }
        _decimationPhase = (_decimationPhase + 1) % _decimationFactor;
    }

    const auto& freq = _transformer.process(_decimated);
    utils::getPowerSpectrum(freq, _spectrum);
    std::transform(_spectrum.begin(), _spectrum.end(), _spectrum.begin(),
                   [](float power) { return utils::FastDb(power); });

    // Put the noise floor at 0, which is what the comb support reads levels against. The
    // in-range path whitens with a cepstral envelope; here a single level per frame is enough,
    // because the comb support only ever compares points a few bins apart, and the median over
    // the band is a robust estimate of it (partials occupy few of those bins).
    _floorScratch.assign(_spectrum.begin() + 1, _spectrum.begin() + _floorBandEnd);
    const auto median = _floorScratch.begin() + _floorScratch.size() / 2;
    std::nth_element(_floorScratch.begin(), median, _floorScratch.end());
    const auto floor = *median + floorMarginDb;
    std::transform(_spectrum.begin(), _spectrum.begin() + _spectrum.size() / 2 + 1,
                   _spectrum.begin(), [floor](float db) { return db - floor; });

    // Own keys, so the in-range path's log (same names, different rate and FFT size) stays
    // intact; a no-op except on the one frame being recorded.
    _logger.Log(_rate, "lowBandRate");
    _logger.Log(_transformer.fftSize(), "lowBandFftSize");
    _logger.Log(floor, "lowBandFloorDb");
    _logger.Log(_spectrum.data(), _spectrum.size(), "lowBandSpectrum");
}

LowBandAnalyzer::Verdict LowBandAnalyzer::below(float inRangeEstimate,
                                                Diagnostics* diagnostics) const {
    if (diagnostics != nullptr) {
        diagnostics->clear();
    }
    if (inRangeEstimate <= 0.f) {
        return {};
    }
    const auto lastIndex = static_cast<float>(_spectrum.size() / 2);
    Verdict best;
    auto bestProminence = 0.f;
    auto bestHarmonicOrder = 0;  // the winning candidate's p, for the diagnostics below
    // Which fundamental in the octave below the range explains the band best? The candidates
    // are the fundamentals the in-range estimate could be a partial of - it at small rational
    // ratios q/p. Confining the search to those *is* the statement that the two frequencies
    // belong to one string: mains hum stands at no such ratio to the note it happens to
    // accompany, and a candidate that fits only the hum would otherwise leave every partial it
    // finds unexplained by the estimate, which reads as maximal evidence for exactly the wrong
    // conclusion. Every candidate is weighed over the same number of harmonics, so none is
    // favoured for merely having more of them within reach.
    for (auto p = 2; p <= _maxHarmonic; ++p) {
        for (auto q = 1; q <= std::min(maxRatioNumerator, p - 1); ++q) {
            if (std::gcd(p, q) != 1) {
                continue;
            }
            const auto candidate = inRangeEstimate * q / p;
            if (candidate < _minFrequency || candidate >= _rangeFloor) {
                continue;
            }
            const auto f0Index = candidate / _binFreq;
            auto explained = 0.f;
            auto unexplained = 0.f;
            auto explainedCount = 0;
            auto unexplainedCount = 0;
            for (auto m = 1; m <= subHarmonicCombSize; ++m) {
                const auto index = m * f0Index;
                if (index + f0Index / 2.f >= lastIndex) {
                    break;
                }
                const auto prominence = spectralProminence(_spectrum, index, f0Index / 2.f);
                if (isEstimateHarmonic(m, p)) {
                    explained += prominence;
                    ++explainedCount;
                } else {
                    unexplained += prominence;
                    ++unexplainedCount;
                }
            }
            // Per member, not in total: how many of the comb's harmonics the estimate explains is
            // a property of the ratio between the two frequencies, not of the evidence, so a
            // total would score the same physical case differently for p = 2 and p = 3.
            const auto support =
                combSupport(explained, explainedCount, unexplained, unexplainedCount);
            const auto total = explained + unexplained;
            if (diagnostics != nullptr) {
                diagnostics->candidateHz.push_back(candidate);
                diagnostics->candidateSupport.push_back(support);
                diagnostics->candidateProminence.push_back(total);
            }
            // The fit itself goes on total prominence: how much of the band the candidate's comb
            // accounts for at all, which is what makes it the best-fitting one rather than the
            // one whose harmonics happen to be most lopsided.
            if (total > bestProminence) {
                bestProminence = total;
                best = {candidate, support};
                bestHarmonicOrder = p;
            }
        }
    }
    if (diagnostics != nullptr && best.frequency > 0.f) {
        // Re-walk the winner's comb, this time recording it rather than summing it.
        const auto f0Index = best.frequency / _binFreq;
        for (auto m = 1; m <= subHarmonicCombSize; ++m) {
            const auto index = m * f0Index;
            if (index + f0Index / 2.f >= lastIndex) {
                break;
            }
            diagnostics->combHz.push_back(index * _binFreq);
            diagnostics->combProminence.push_back(
                spectralProminence(_spectrum, index, f0Index / 2.f));
            diagnostics->combExplained.push_back(isEstimateHarmonic(m, bestHarmonicOrder) ? 1.f
                                                                                          : 0.f);
        }
    }
    return best;
}
}  // namespace saint
