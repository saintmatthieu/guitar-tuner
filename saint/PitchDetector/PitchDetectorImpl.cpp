#include "PitchDetectorImpl.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "PitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
constexpr auto cutoffFreq = 1500;

constexpr float defaultLeastFrequencyToDetect = /* A0 */ 27.5f;

int getFftOrder(int windowSize) {
    return static_cast<int>(ceilf(log2f((float)windowSize)));
}

int getFftSizeSamples(int windowSize) {
    constexpr auto zeroPadding = 2;
    return 1 << (getFftOrder(windowSize) + zeroPadding);
}

float pitchToFrequency(const Pitch& pitch) {
    if (PitchClass::OneKiloHz == pitch.pitchClass) {
        return 1000 * (1 << pitch.octave);
    }
    const std::unordered_map<PitchClass, int> semitonesFromA{
        {PitchClass::C, -9},  {PitchClass::Db, -8}, {PitchClass::D, -7},  {PitchClass::Eb, -6},
        {PitchClass::E, -5},  {PitchClass::F, -4},  {PitchClass::Gb, -3}, {PitchClass::G, -2},
        {PitchClass::Ab, -1}, {PitchClass::A, 0},   {PitchClass::Bb, 1},  {PitchClass::B, 2},
    };
    const int semitonesFromA4 = semitonesFromA.at(pitch.pitchClass) + (pitch.octave - 4) * 12;
    return 440.f * std::pow(2.f, semitonesFromA4 / 12.f);
}

float getMinFreq(const std::optional<PitchDetector::Config>& config) {
    return config && config->lowestPitch.has_value() ? pitchToFrequency(*config->lowestPitch)
                                                     : defaultLeastFrequencyToDetect;
}

float getMaxFreq(const std::optional<PitchDetector::Config>& config) {
    return config && config->highestPitch.has_value() ? pitchToFrequency(*config->highestPitch)
                                                      : 2000.f;
}

int getWindowSizeSamples(int sampleRate, utils::WindowType windowType,
                         const std::optional<PitchDetector::Config>& config) {
    const auto minFreq = getMinFreq(config);
    const auto numPeriods = utils::windowOrders.at(static_cast<size_t>(windowType)) * 2 + 1.3;
    const auto windowSizeMs = 1000. * numPeriods / minFreq;
    return static_cast<int>(windowSizeMs * sampleRate / 1000);
}

void applyWindow(const std::vector<float>& window, std::vector<float>& input) {
    for (auto i = 0u; i < window.size(); ++i) {
        input[i] *= window[i];
    }
}

void getXCorr(RealFft& fft, std::vector<float>& time, std::vector<std::complex<float>> freq,
              const std::vector<float>& lpWindow) {
    auto timeData = time.data();

    for (auto i = 0u; i < lpWindow.size(); ++i) {
        auto& X = freq[i];
        X *= lpWindow[i] * std::complex<float>{X.real(), -X.imag()};
    }
    std::fill(freq.data() + lpWindow.size(), freq.data() + fft.size / 2, 0.f);
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
    const int cutoffBin = std::min(fftSize / 2, fftSize * cutoffFreq / sampleRate);
    const int rollOffSize = fftSize * 200 / sampleRate;
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

std::optional<int> takeThisIndexInstead(const std::vector<float>& cepstrum, int leftmost,
                                        int maxValuedIndex) {
    // Although we do "low-pass lifter" the cepstrum to reduce the risk of confusing the first
    // harmonic for the fundamental, it still happens. Liftering some introduces damage, according
    // to benchmarking.
    // Let's check if there is a peak near maxValuedIndex / 2 and if it rivals that at
    // maxValuedIndex in terms of value. If it does, let's go for this one.
    auto halfIndex = maxValuedIndex / 2;

    if (halfIndex < leftmost) {
        return {};
    }

    // climb to the top
    while (halfIndex + 1 < cepstrum.size() && cepstrum[halfIndex + 1] >= cepstrum[halfIndex]) {
        ++halfIndex;
    }
    while (halfIndex - 1 >= 0 && cepstrum[halfIndex - 1] > cepstrum[halfIndex]) {
        --halfIndex;
    }
    if (const auto tooFar = std::abs(maxValuedIndex / 2 - halfIndex) > 1) {
        return {};
    }

    const auto ratio = cepstrum[halfIndex] / cepstrum[maxValuedIndex];
    if (ratio > 0.78f) {
        return halfIndex;
    } else {
        return {};
    }
}

std::vector<std::pair<int, int>> primeCombinations(int K) {
    // If K == 4, returns {(1,2), (1,3), (1,4), (2,3), (3,4)}, i.e., all tuples with elements in [1,
    // K] whose GCD is 1.
    std::vector<std::pair<int, int>> combinations;
    for (auto i = 1; i <= K; ++i) {
        for (auto j = i + 1; j <= K; ++j) {
            if (std::gcd(i, j) == 1) {
                combinations.emplace_back(i, j);
            }
        }
    }
    return combinations;
}

float disambiguateFundamentalIndex(float priorIndex, const std::vector<float>& idealSpectrum,
                                   float minF0) {
    const auto& spec = idealSpectrum;

    constexpr auto K = 15;  // Number of harmonics used

    constexpr auto numAlternatives = 5;
    constexpr std::array<float, numAlternatives> alternatives{
        1.f,            // Keep 1 in front in case all estimates are zero
        2.f, 3.f, 4.f,  //
        1 / 2.f};

    std::vector<float> scores(numAlternatives, 0.f);
    std::vector<std::array<float, K>> indicesVector(numAlternatives);
    std::vector<std::pair<int, int>> combinations = primeCombinations(K);

    std::array<int, K> harmonicNumbers;
    std::iota(harmonicNumbers.begin(), harmonicNumbers.end(), 1);

    for (auto d = 0; d < numAlternatives; ++d) {
        const auto multiplier = alternatives[d];
        const auto f0 = priorIndex * multiplier;

        if (f0 < minF0) {
            scores[d] = 0.f;
            continue;
        }

        auto& indices = indicesVector[d];
        std::vector<float> peakValues(K);

        for (const auto k : harmonicNumbers) {
            const auto fk = k * f0;
            const auto peakIndex =
                utils::getIndexOfClosestLocalMaximum(spec, static_cast<int>(fk + .5f));
            const float fractionalPart = utils::quadFit(spec.data() + peakIndex - 1);
            peakValues[k - 1] = std::max(spec[peakIndex], 0.f);
            indices[k - 1] = peakIndex + fractionalPart;
        }

        std::unordered_set<float> uniqueIndices(indices.begin(), indices.end());
        if (uniqueIndices.size() < indices.size()) {
            // Duplicates - search is degenerate. Break.
            scores[d] = 0.f;
            break;
        }

        if (std::all_of(peakValues.begin(), peakValues.end(), [](float w) { return w == 0.f; })) {
            scores[d] = 0.f;
            continue;
        }

        std::vector<float> perceptualProducts(combinations.size());
        std::transform(combinations.begin(), combinations.end(), perceptualProducts.begin(),
                       [&peakValues](const std::pair<int, int>& combination) {
                           const auto& [k1, k2] = combination;
                           return peakValues[k1 - 1] * peakValues[k2 - 1];
                       });

        const auto sum = std::accumulate(perceptualProducts.begin(), perceptualProducts.end(), 0.f);
        scores[d] = sum;
    }

    const auto it = std::max_element(scores.begin(), scores.end());
    const auto d = std::distance(scores.begin(), it);
    const auto a = utils::leastSquareFit(harmonicNumbers, indicesVector[d]);

    return a;
}
}  // namespace

float PitchDetectorImpl::disambiguateEstimate(float priorEstimate,
                                              const std::vector<float>& idealSpectrum) const {
    const auto priorIndex = priorEstimate / _binFreq;
    const auto minF0 = _minFreq / _binFreq;
    return disambiguateFundamentalIndex(priorIndex, idealSpectrum, minF0) * _binFreq;
}

float PitchDetectorImpl::getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const {
    const auto cepstrum = cepstrumData.vec();

    // We're using this for a tuner, so look between 30Hz (a detuned E0 on a bass
    // guitar) and 500Hz (Ukulele high A is 440Hz)
    const auto maxPeriod = 1. / _minFreq;
    const auto minPeriod = 1. / _maxFreq;
    const auto leftmost = static_cast<int>(minPeriod * _sampleRate);
    const auto rightmost = std::min<int>(maxPeriod * _sampleRate, cepstrum.size());
    const auto it = std::max_element(cepstrum.begin() + leftmost, cepstrum.begin() + rightmost);
    auto maxCepstrumIndex = std::distance(cepstrum.begin(), it);
    if (maxCepstrumIndex == 0)
        return 0.f;
    else if (maxCepstrumIndex == cepstrum.size())
        return static_cast<float>(_sampleRate) / maxCepstrumIndex;

    const auto bestIndex =
        takeThisIndexInstead(cepstrum, leftmost, maxCepstrumIndex).value_or(maxCepstrumIndex);
    return static_cast<float>(_sampleRate) / bestIndex;
}

PitchDetectorImpl::PitchDetectorImpl(int sampleRate, ChannelFormat channelFormat,
                                     int samplesPerBlockPerChannel,
                                     const std::optional<Config>& config,
                                     std::unique_ptr<PitchDetectorLoggerInterface> logger)
    : _sampleRate(sampleRate),
      _channelFormat(channelFormat),
      _blockSize(samplesPerBlockPerChannel),
      _logger(std::move(logger)),
      _windowType(utils::WindowType::MinimumThreeTerm),
      _window(utils::getAnalysisWindow(getWindowSizeSamples(sampleRate, _windowType, config),
                                       _windowType)),
      _fftSize(getFftSizeSamples(static_cast<int>(_window.size()))),
      _binFreq(static_cast<float>(sampleRate) / _fftSize),
      _fwdFft(_fftSize),
      _cepstrumFft(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _minFreq(getMinFreq(config)),
      _maxFreq(getMaxFreq(config)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / _minFreq))),
      _windowXcor(getWindowXCorr(_fwdFft, _window, _lpWindow)),
      _latencySamples(std::max(static_cast<int>(_window.size()) - samplesPerBlockPerChannel, 0)),
      _audioBuffer(_latencySamples, 0.f),
      _maxAlpha(0.0015f * sampleRate / samplesPerBlockPerChannel) {
    //
    _audioBuffer.reserve(_window.size());
    _logger->SamplesRead(-_latencySamples);
}

float PitchDetectorImpl::refineEstimateBasedOnStrongestHarmonic(
    const std::vector<float>& logSpectrum, float targetFreq) const {
    constexpr auto numHarmonics = 10;
    std::vector<int> peakBins(numHarmonics);
    for (auto k = 1; k <= numHarmonics; ++k) {
        const auto targetBin = static_cast<int>(targetFreq * k / _binFreq + 0.5f);
        const auto peakBin = utils::getIndexOfClosestLocalMaximum(logSpectrum, targetBin);
        peakBins[k - 1] = peakBin;
    }
    const auto it = std::max_element(peakBins.begin(), peakBins.end(),
                                     [&](int i, int j) { return logSpectrum[i] < logSpectrum[j]; });
    const int k = std::distance(peakBins.begin(), it) + 1;
    const int peakBin{*it};  // make sure there is no narrowing conversion

    // parabolic interpolation
    if (peakBin <= 0 || peakBin >= logSpectrum.size() - 1) {
        return static_cast<float>(peakBin) * _binFreq / k;
    }

    const auto delta = utils::quadFit(&logSpectrum[peakBin - 1]);

    const auto refined = (static_cast<float>(peakBin) + delta) * _binFreq / k;
    return refined;
}

float PitchDetectorImpl::process(const float* audio, float* presenceScore) {
    // Append new audio samples to buffer
    if (_channelFormat == ChannelFormat::Mono) {
        _audioBuffer.insert(_audioBuffer.end(), audio, audio + _blockSize);
    } else {
        assert(_channelFormat == ChannelFormat::Stereo);
        for (auto i = 0; i < _blockSize; ++i) {
            const auto mix = 0.5f * (audio[i * 2] + audio[i * 2 + 1]);
            _audioBuffer.push_back(mix);
        }
    }

    assert(_audioBuffer.size() >= _window.size());
    if (_audioBuffer.size() < _window.size()) {
        if (!_bufferErrorLoggedAlready) {
            _bufferErrorLoggedAlready = true;
            std::cerr << "PitchDetectorImpl::process called before enough samples were read\n";
        }
        return 0.f;
    }

    std::vector<float> time(_fftSize);
    _logger->StartNewEstimate();

    // Copy the most recent window of samples
    const auto bufferStart = _audioBuffer.end() - _window.size();
    std::copy(bufferStart, _audioBuffer.end(), time.begin());

    // Remove old samples, keeping only what's needed for the next window
    const auto samplesToKeep = _window.size() - _blockSize;
    _audioBuffer.erase(_audioBuffer.begin(), _audioBuffer.end() - samplesToKeep);
    std::fill(time.begin() + _window.size(), time.begin() + _fftSize, 0.f);

    _logger->SamplesRead(_blockSize);
    _logger->Log(_sampleRate, "sampleRate");
    _logger->Log(_fftSize, "fftSize");
    _logger->Log(time.data(), time.size(), "inputAudio");

    // zero all samples below -60dB
    std::for_each(time.begin(), time.end(), [](float& x) {
        constexpr float threshold = 0.001f;
        if (std::abs(x) < threshold) {
            x = 0.f;
        }
    });

    applyWindow(_window, time);
    _logger->Log(time.data(), time.size(), "windowedAudio");

    // Forward FFT
    std::vector<std::complex<float>> freq = getSpectrum(_fwdFft, time.data());
    _logger->Log(freq.data(), freq.size(), "spectrum",
                 [](const std::complex<float>& X) { return std::abs(X); });

    std::vector<float> dbSpectrum(_fftSize);
    utils::getPowerSpectrum(freq, dbSpectrum);
    std::transform(dbSpectrum.begin(), dbSpectrum.end(), dbSpectrum.begin(),
                   [](float power) { return utils::FastDb(power); });
    assert(utils::isSymmetric(dbSpectrum));
    _logger->Log(dbSpectrum.data(), dbSpectrum.size(), "dbSpectrum");

    // Compute cross-correlation
    getXCorr(_fwdFft, time, freq, _lpWindow);
    _logger->Log(time.data(), time.size(), "xcorr");
    _logger->Log(time.data(), time.size(), "xcorrFlattened",
                 [this, i = 0](float x) mutable { return x / std::max(_windowXcor[i++], 1e-6f); });

    utils::Finally finally([this]() { _logger->EndNewEstimate(nullptr, 0); });

    auto maxIndex = 0;
    auto wentNegative = false;
    auto maximum = 0.f;
    for (auto i = 0; i < _lastSearchIndex; ++i) {
        wentNegative |= time[i] < 0;
        if (wentNegative && time[i] > maximum) {
            maximum = time[i];
            maxIndex = i;
        }
    }

    maximum /= _windowXcor[maxIndex];
    if (presenceScore) {
        *presenceScore = maximum;
    }

    Aligned<std::vector<float>> cepstrumAligned;
    toCepstrum(dbSpectrum, _cepstrumFft, cepstrumAligned);

    updateNoiseProfile(dbSpectrum, cepstrumAligned.value, maximum);

    if (maximum < _threshold) {
        return 0.f;
    }

    const auto xcorrEstimate = static_cast<float>(_sampleRate) / maxIndex;

    const auto refinedEstimate = refineEstimateBasedOnStrongestHarmonic(dbSpectrum, xcorrEstimate);

    auto idealSpectrum = dbSpectrum;
    toIdealSpectrum(idealSpectrum, cepstrumAligned.value);

    const auto disambiguatedEstimate = disambiguateEstimate(refinedEstimate, idealSpectrum);

    _logger->EndNewEstimate(nullptr, 0);

    return disambiguatedEstimate;
}

void PitchDetectorImpl::updateNoiseProfile(const std::vector<float>& dbSpectrum,
                                           const std::vector<float>& cepstrum,
                                           float presenceScore) {
    // Adaptation rate depends on presence score:
    // - presenceScore = 0 -> alpha = maxAlpha (fast adaptation, confident it's noise)
    // - presenceScore = threshold -> alpha = 0 (no adaptation, pitch is present)

    // Linear interpolation: alpha decreases as presence score increases
    const float alpha = std::max(0.f, _maxAlpha * (1.f - presenceScore / _threshold));

    if (alpha == 0.f) {
        return;  // No adaptation when pitch is detected
    }

    std::vector<float> lifteredCepstrum = cepstrum;
    const auto cutoffIndex = std::min<int>(_sampleRate / 2500.f, cepstrum.size());
    std::fill(lifteredCepstrum.begin() + cutoffIndex, lifteredCepstrum.end() - cutoffIndex + 1,
              0.f);

    const std::vector<float> envelope = fromCepstrum(_cepstrumFft, lifteredCepstrum.data());

    if (_noiseProfile.empty()) {
        _noiseProfile = envelope;
    } else {
        for (size_t i = 0; i < _noiseProfile.size(); ++i) {
            _noiseProfile[i] = alpha * envelope[i] + (1.f - alpha) * _noiseProfile[i];
        }
    }
}

void PitchDetectorImpl::toIdealSpectrum(std::vector<float>& logSpectrum,
                                        const std::vector<float>& cepstrum) {
    auto& spec = logSpectrum;

    std::vector<float> lifteredCepstrum = cepstrum;
    const auto cutoffIndex = std::min<int>(_sampleRate / 2500.f, cepstrum.size());
    std::fill(lifteredCepstrum.begin() + cutoffIndex, lifteredCepstrum.end() - cutoffIndex + 1,
              0.f);

    const std::vector<float> spectrumEnvelope = fromCepstrum(_cepstrumFft, lifteredCepstrum.data());
    _logger->Log(spectrumEnvelope.data(), spectrumEnvelope.size(), "spectrumEnvelope");

    // If we have a noise profile, use it; otherwise fall back to current envelope
    const std::vector<float>& profileToSubtract =
        _noiseProfile.empty() ? spectrumEnvelope : _noiseProfile;

    std::transform(spec.begin(), spec.end(), profileToSubtract.begin(), spec.begin(),
                   std::minus<float>());

    // Calculate the variance from 5kHz to the Nyquist
    const auto minFreq = 5000.f;
    const auto minBin = static_cast<int>(minFreq / _binFreq);
    const auto N = static_cast<float>(static_cast<int>(spec.size()) - minBin);

    // Expected value E
    const auto E = std::accumulate(spec.begin() + minBin, spec.end(), 0.f,
                                   [](float acc, float val) { return acc + val; }) /
                   N;

    const auto variance =
        std::accumulate(spec.begin(), spec.end(), 0.f,
                        [E](float acc, float val) { return acc + (val - E) * (val - E); }) /
        N;

    const auto stdDev = std::sqrt(variance);
    const auto noiseThreshold = stdDev * 1.5f;

    std::transform(spec.begin(), spec.end(), spec.begin(),
                   [noiseThreshold](float x) { return x - noiseThreshold; });

    _logger->Log(spec.data(), spec.size(), "idealSpectrum");

    assert(utils::isSymmetric(spec));
    assert(utils::isPowerOfTwo(spec.size()));
}

}  // namespace saint
