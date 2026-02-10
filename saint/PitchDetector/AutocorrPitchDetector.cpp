#include "AutocorrPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "AutocorrPitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
constexpr auto cutoffFreq = 1500;

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
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / minFreq))),
      _windowXcor(getWindowXCorr(_fwdFft, fftWindow, _lpWindow)) {}

float AutocorrPitchDetector::process(const std::vector<std::complex<float>>& freq,
                                     float* presenceScore) {
    std::vector<float> time(_fftSize);

    // Compute cross-correlation
    getXCorr(_fwdFft, time, freq, _lpWindow);
    _logger.Log(time.data(), time.size(), "xcorr");

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

    constexpr auto threshold = 0.88f;
    if (maximum < threshold) {
        return 0.f;
    } else {
        return static_cast<float>(_sampleRate) / maxIndex;
    }
}
}  // namespace saint
