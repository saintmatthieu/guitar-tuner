#include "AutocorrPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "AutocorrPitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
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
                                             PitchDetectorLoggerInterface& logger)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _fwdFft(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / minFreq))),
      _windowXcorr(getWindowXCorr(_fwdFft, fftWindow, _lpWindow)) {}

float AutocorrPitchDetector::process(const std::vector<std::complex<float>>& freq,
                                     float* presenceScore, std::optional<float> constraint) {
    std::vector<float> xcorr(_fftSize);
    _logger.Log(_windowXcorr.data(), _windowXcorr.size(), "windowXcorr");

    // Compute cross-correlation
    getXCorr(_fwdFft, xcorr, freq, _lpWindow);
    _logger.Log(xcorr.data(), xcorr.size(), "xcorr");

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
        wentNegative |= xcorr[i] < 0;
        if (wentNegative && i >= firstSearchIndex && xcorr[i] > maximum) {
            maximum = xcorr[i];
            maxIndex = i;
        }
    }

    maximum /= _windowXcorr[maxIndex];
    if (presenceScore) {
        *presenceScore = maximum;
    }

    const auto fracIndex = utils::quadFit(&xcorr[maxIndex - 1]);
    const auto refinedIndex = maxIndex + fracIndex;

    return maxIndex == 0 ? 0.f : static_cast<float>(_sampleRate) / refinedIndex;
}
}  // namespace saint
