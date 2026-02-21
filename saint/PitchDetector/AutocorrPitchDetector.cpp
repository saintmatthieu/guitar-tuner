#include "AutocorrPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "AutocorrPitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
constexpr auto cutoffFreq = 1500;

std::vector<float> upsampleXCorr(std::vector<float> xcorr, int sampleRate) {
    Upsampler upsampler(sampleRate);
    std::rotate(xcorr.begin(), xcorr.begin() + xcorr.size() / 2, xcorr.end());
    std::vector<float> upXcorr = upsampler.process(xcorr.data(), xcorr.size());
    std::rotate(upXcorr.begin(), upXcorr.begin() + upXcorr.size() / 2 + upsampler.delaySamples(),
                upXcorr.end());
    return upXcorr;
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
}  // namespace

AutocorrPitchDetector::AutocorrPitchDetector(int sampleRate, int fftSize,
                                             const std::vector<float>& fftWindow, float minFreq,
                                             PitchDetectorLoggerInterface& logger)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _fwdFft(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / minFreq)) *
                       Upsampler::factor),
      _windowXcorr(upsampleXCorr(getWindowXCorr(_fwdFft, fftWindow, _lpWindow), sampleRate)) {}

float AutocorrPitchDetector::process(const std::vector<std::complex<float>>& freq,
                                     float* presenceScore, std::optional<float> constraint) {
    std::vector<float> xcorr(_fftSize);
    _logger.Log(_windowXcorr.data(), _windowXcorr.size(), "windowXcorr");

    // Compute cross-correlation
    getXCorr(_fwdFft, xcorr, freq, _lpWindow);
    _logger.Log(xcorr.data(), xcorr.size(), "xcorr");

    const auto upXcorr = upsampleXCorr(std::move(xcorr), _sampleRate);
    _logger.Log(upXcorr.data(), upXcorr.size(), "upXcorr");

    // Determine search range based on constraint
    // A major third is a ratio of 5/4 = 1.26 (up) or 4/5 = 0.8 (down)
    constexpr auto majorThirdRatio = 1.26f;
    int firstSearchIndex = 0;
    int lastSearchIndex = _lastSearchIndex;
    const auto upSampleRate = _sampleRate * Upsampler::factor;

    if (constraint.has_value() && constraint.value() > 0.f) {
        const auto constraintFreq = constraint.value();
        const auto minFreq = constraintFreq / majorThirdRatio;
        const auto maxFreq = constraintFreq * majorThirdRatio;
        // Convert frequencies to lag indices (frequency = sampleRate / lag)
        // Higher frequency means smaller lag
        firstSearchIndex = std::max(0, static_cast<int>(upSampleRate / maxFreq));
        lastSearchIndex = std::min(_lastSearchIndex, static_cast<int>(upSampleRate / minFreq) + 1);
    }

    auto maxIndex = 0;
    auto wentNegative = false;
    auto maximum = 0.f;
    for (auto i = 0; i < lastSearchIndex; ++i) {
        wentNegative |= upXcorr[i] < 0;
        if (wentNegative && i >= firstSearchIndex && upXcorr[i] > maximum) {
            maximum = upXcorr[i];
            maxIndex = i;
        }
    }

    maximum /= _windowXcorr[maxIndex];
    if (presenceScore) {
        *presenceScore = maximum;
    }

    return maxIndex == 0 ? 0.f : static_cast<float>(upSampleRate) / maxIndex;
}
}  // namespace saint
