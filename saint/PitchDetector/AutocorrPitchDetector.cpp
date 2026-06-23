#include "AutocorrPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "AutocorrPitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
// Zero-phase band-pass applied to the (real, low-passed) power spectrum in `freq[0, numBins)`:
// keep [loBin, hiBin], with a linear ramp of `rollBins` on each side, zero beyond. A no-op when
// loBin < 0. This is a plain per-call spectral mask (no state, no feedback), so the only thing
// that changes block-to-block is which bins are kept - it has no stability dynamics.
void applyBandPass(std::vector<std::complex<float>>& freq, int numBins, int loBin, int hiBin,
                   int rollBins) {
    if (loBin < 0) {
        return;
    }
    const auto loStop = std::max(0, loBin - rollBins);
    for (auto i = 0; i < loStop; ++i) {
        freq[i] = 0.f;
    }
    for (auto i = loStop; i < loBin; ++i) {
        freq[i] *= static_cast<float>(i - loStop) / rollBins;
    }
    const auto hiStop = std::min(numBins, hiBin + rollBins);
    for (auto i = hiBin + 1; i < hiStop; ++i) {
        freq[i] *= 1.f - static_cast<float>(i - hiBin) / rollBins;
    }
    for (auto i = hiStop; i < numBins; ++i) {
        freq[i] = 0.f;
    }
}

void getXCorr(RealFft& fft, std::vector<float>& time, std::vector<std::complex<float>>& freq,
              const std::vector<float>& lpWindow, int bpLoBin = -1, int bpHiBin = 0,
              int bpRollBins = 0) {
    auto timeData = time.data();

    for (auto i = 0u; i < lpWindow.size(); ++i) {
        auto& X = freq[i];
        X *= lpWindow[i] * std::complex<float>{X.real(), -X.imag()};
    }
    std::fill(freq.data() + lpWindow.size(), freq.data() + fft.size / 2, 0.f);
    applyBandPass(freq, static_cast<int>(lpWindow.size()), bpLoBin, bpHiBin, bpRollBins);
    fft.inverse(freq.data(), timeData);
    if (timeData[0] < 1e-6f) {
        return;
    }
    const auto normalizer = 1.f / timeData[0];
    for (auto i = 0; i < fft.size; ++i) {
        timeData[i] *= normalizer;
    }
}

std::vector<float> getLpWindow(int sampleRate, int fftSize) {
    std::vector<float> window(fftSize / 2);
    const int cutoffBin = std::min(fftSize / 2, fftSize * autocorrCutoffFreqHz / sampleRate);
    const int rollOffSize = fftSize * autocorrRolloffHz / sampleRate;
    std::fill(window.begin(), window.begin() + cutoffBin, 1.f);
    for (auto i = 0; i < rollOffSize && cutoffBin + rollOffSize < fftSize / 2; ++i) {
        window[cutoffBin + i] = 1.f - i / static_cast<float>(rollOffSize);
    }
    std::fill(window.begin() + cutoffBin + rollOffSize, window.end(), 0.f);
    return window;
}

std::vector<std::complex<float>> getSpectrum(RealFft& fft, const float* timeData) {
    Aligned<std::vector<std::complex<float>>> freqAligned;
    auto& freq = freqAligned.value;
    freq.resize(fft.size / 2);
    fft.forward(timeData, freq.data());
    return freqAligned.value;
}

std::vector<float> getWindowXCorr(RealFft& fft, const std::vector<float>& window,
                                  const std::vector<float>& lpWindow) {
    Aligned<std::vector<float>> xcorrAligned;
    auto& xcorr = xcorrAligned.value;
    xcorr.resize(fft.size);
    std::copy(window.begin(), window.end(), xcorr.begin());
    std::fill(xcorr.begin() + window.size(), xcorr.end(), 0.f);
    std::vector<std::complex<float>> freq = getSpectrum(fft, xcorr.data());
    getXCorr(fft, xcorr, freq, lpWindow);
    return xcorr;
}
}  // namespace

AutocorrPitchDetector::AutocorrPitchDetector(int sampleRate, int fftSize,
                                             const std::vector<float>& fftWindow, float minFreq,
                                             PitchDetectorLoggerInterface& logger,
                                             bool applyConstraintBandPass)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _applyConstraintBandPass(applyConstraintBandPass),
      _fwdFft(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / minFreq))),
      _windowXcorr(getWindowXCorr(_fwdFft, fftWindow, _lpWindow)),
      _xcorr(static_cast<size_t>(_fftSize), 0.f),
      _freqScratch(static_cast<size_t>(_fftSize) / 2) {
    if (autocorrAveragingFrameCount > 1) {
        _xcorrHistory.assign(autocorrAveragingFrameCount, std::vector<float>(_fftSize, 0.f));
        _averagedXcorr.resize(_fftSize, 0.f);
    }
}

void AutocorrPitchDetector::reset() {
    _historyWritePos = 0;
    _historyFilled = 0;
}

const std::vector<float>& AutocorrPitchDetector::averageOverFrames(
    const std::vector<float>& xcorr) {
    if (autocorrAveragingFrameCount <= 1) {
        return xcorr;
    }

    // Overwrite the oldest slot (equal-size copy, no reallocation in the RT path).
    _xcorrHistory[_historyWritePos] = xcorr;
    _historyWritePos = (_historyWritePos + 1) % autocorrAveragingFrameCount;
    if (_historyFilled < autocorrAveragingFrameCount) {
        ++_historyFilled;
    }

    // Average the valid frames. While the ring is filling, only [0, _historyFilled)
    // hold real data; once full, _historyFilled == K and every slot is valid.
    std::fill(_averagedXcorr.begin(), _averagedXcorr.end(), 0.f);
    for (auto k = 0; k < _historyFilled; ++k) {
        const auto& frame = _xcorrHistory[k];
        for (auto i = 0u; i < _averagedXcorr.size(); ++i) {
            _averagedXcorr[i] += frame[i];
        }
    }
    const auto scale = 1.f / _historyFilled;
    for (auto& v : _averagedXcorr) {
        v *= scale;
    }
    return _averagedXcorr;
}

namespace {
// Parabolic-refined location and height of the largest autocorrelation peak in [lagLo, lagHi].
struct AcfPeak {
    float lag;
    float height;
};
AcfPeak refinedPeakIn(const std::vector<float>& acf, int lagLo, int lagHi) {
    // Clamp both ends into [1, size-2] so the search and the quadFit neighbour reads stay in
    // bounds even when the caller's target lag is garbage (e.g. a degenerate fundamental peak).
    const auto maxLag = static_cast<int>(acf.size()) - 2;
    lagLo = std::max(1, std::min(lagLo, maxLag));
    lagHi = std::max(lagLo, std::min(lagHi, maxLag));
    auto mi = lagLo;
    auto mx = acf[lagLo];
    for (auto i = lagLo + 1; i <= lagHi; ++i) {
        if (acf[i] > mx) {
            mx = acf[i];
            mi = i;
        }
    }
    // Parabolic sub-sample offset, guarded: a flat/degenerate peak makes quadFit blow up (or
    // return NaN), and an out-of-range lag would later index _windowXcorr out of bounds. A true
    // peak's offset is within +/-1 sample, so clamp anything else back to the integer max.
    auto frac = utils::quadFit(&acf[mi - 1]);
    if (!(frac > -1.f && frac < 1.f)) {
        frac = 0.f;
    }
    return {mi + frac, mx};
}
}  // namespace

float AutocorrPitchDetector::process(const std::vector<std::complex<float>>& freq,
                                     float* presenceScore, std::optional<float> constraint,
                                     float* harmonicConsistencyCents, float* secondaryPeakScore) {
    _logger.Log(_windowXcorr.data(), _windowXcorr.size(), "windowXcorr");

    // When locked, band-pass the autocorrelation to a window around the constrained fundamental
    // (see autocorrConstraintBandHalfWidthSemitones). The band tracks the constraint each block.
    // It is only applied when it spans enough FFT bins to be resolvable: at low fundamentals a
    // few-semitone band is only a couple of bins wide, which would quantise the estimate, so those
    // notes (which don't suffer the sub-fundamental contamination anyway) stay broadband.
    constexpr int kMinBandBins = 5;
    int bpLoBin = -1, bpHiBin = 0, bpRollBins = 0;
    if (_applyConstraintBandPass && constraint.has_value() && constraint.value() > 0.f &&
        autocorrConstraintBandHalfWidthSemitones > 0.f) {
        const auto ratio = std::exp2(autocorrConstraintBandHalfWidthSemitones / 12.f);
        const auto toBin = [this](float f) {
            return static_cast<int>(f * _fftSize / static_cast<float>(_sampleRate));
        };
        const auto loBin = toBin(constraint.value() / ratio);
        const auto hiBin = std::min(_fftSize / 2 - 1, toBin(constraint.value() * ratio) + 1);
        if (hiBin - loBin >= kMinBandBins) {
            // Engage only when sub-fundamental energy (what pulls the ACF peak flat) is significant
            // relative to the in-band energy - i.e. in the contaminated decay tail, not in a healthy
            // note (see autocorrConstraintBandContaminationRatio).
            const auto subLoBin = std::max(1, toBin(autocorrSubFundamentalFloorHz));
            auto eBand = 0.f;
            for (auto i = loBin; i <= hiBin; ++i) {
                eBand += std::norm(freq[i]);
            }
            auto eSub = 0.f;
            for (auto i = subLoBin; i < loBin; ++i) {
                eSub += std::norm(freq[i]);
            }
            if (eSub > autocorrConstraintBandContaminationRatio * eBand) {
                bpLoBin = loBin;
                bpHiBin = hiBin;
                bpRollBins = std::max(1, (hiBin - loBin) / 6);
            }
        }
    }

    // Compute cross-correlation. getXCorr overwrites its spectrum argument in place,
    // so work on a reused copy and leave the caller's `freq` untouched (the impl still
    // needs it for the power spectrum).
    std::copy(freq.begin(), freq.end(), _freqScratch.begin());
    getXCorr(_fwdFft, _xcorr, _freqScratch, _lpWindow, bpLoBin, bpHiBin, bpRollBins);
    _logger.Log(_xcorr.data(), _xcorr.size(), "xcorr");

    // Average over the last few frames to suppress noise-driven octave errors
    // (idea #1). `acf` aliases `_xcorr` when averaging is disabled.
    const std::vector<float>& acf = averageOverFrames(_xcorr);
    _logger.Log(acf.data(), acf.size(), "averagedXcorr");

    // Determine search range based on constraint
    int firstSearchIndex = 0;
    int lastSearchIndex = _lastSearchIndex;

    if (constraint.has_value() && constraint.value() > 0.f) {
        const auto constraintFreq = constraint.value();
        const auto minFreq = constraintFreq / majorThirdRatio;
        const auto maxFreq = constraintFreq * majorThirdRatio;
        // Convert frequencies to lag indices (frequency = sampleRate / lag)
        // Higher frequency means smaller lag
        firstSearchIndex = std::max(0, static_cast<int>(_sampleRate / maxFreq));
        lastSearchIndex = std::min(_lastSearchIndex, static_cast<int>(_sampleRate / minFreq) + 1);
    }

    auto maxIndex = 0;
    auto wentNegative = false;
    auto maximum = 0.f;
    for (auto i = 0; i < lastSearchIndex; ++i) {
        wentNegative |= acf[i] < 0;
        if (wentNegative && i >= firstSearchIndex && acf[i] > maximum) {
            maximum = acf[i];
            maxIndex = i;
        }
    }

    maximum /= _windowXcorr[maxIndex];
    if (presenceScore) {
        *presenceScore = maximum;
    }

    if (maxIndex == 0) {
        return 0.f;
    }

    const auto fracIndex = utils::quadFit(&acf[maxIndex - 1]);
    const auto refinedIndex = maxIndex + fracIndex;

    // Harmonic-lag self-consistency: locate the ACF peak near twice the fundamental lag. For a
    // clean periodic signal it sits at exactly 2*refinedIndex (consistency ~0) regardless of the
    // pitch, so a true shift does not trip it; contamination/noise that pulls the fundamental peak
    // leaves the 2L peak behind, so the deviation flags a wandered, untrustworthy estimate.
    if (harmonicConsistencyCents || secondaryPeakScore) {
        const auto target = 2.f * refinedIndex;
        const auto peak = refinedPeakIn(acf, static_cast<int>(target * 0.9f),
                                        std::min(static_cast<int>(target * 1.1f), _fftSize / 2 - 1));
        if (harmonicConsistencyCents) {
            *harmonicConsistencyCents = 1200.f * std::log2(peak.lag / target);
        }
        if (secondaryPeakScore) {
            const auto wlag = std::max(
                0, std::min(static_cast<int>(peak.lag), static_cast<int>(_windowXcorr.size()) - 1));
            *secondaryPeakScore = _windowXcorr[wlag] > 0.f ? peak.height / _windowXcorr[wlag] : 0.f;
        }
    }

    return maxIndex == 0 ? 0.f : static_cast<float>(_sampleRate) / refinedIndex;
}
}  // namespace saint
