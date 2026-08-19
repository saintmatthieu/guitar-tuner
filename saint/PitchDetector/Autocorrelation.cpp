#include "Autocorrelation.h"

#include <algorithm>

namespace saint {
namespace {
std::vector<std::complex<float>> getSpectrum(RealFft& fft, const float* timeData) {
    Aligned<std::vector<std::complex<float>>> freqAligned;
    auto& freq = freqAligned.value;
    freq.resize(fft.size / 2);
    fft.forward(timeData, freq.data());
    return freqAligned.value;
}
}  // namespace

void getXCorr(RealFft& fft, std::vector<float>& time, std::vector<std::complex<float>>& freq,
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

AutocorrPeak findAutocorrPeak(const std::vector<float>& xcorr,
                              const std::vector<float>& windowXcorr, int firstLag, int lastLag) {
    AutocorrPeak peak;
    auto wentNegative = false;
    auto maximum = 0.f;
    for (auto i = 0; i < lastLag; ++i) {
        wentNegative |= xcorr[i] < 0;
        if (wentNegative && i >= firstLag && xcorr[i] > maximum) {
            maximum = xcorr[i];
            peak.lag = i;
        }
    }
    peak.presence = maximum / windowXcorr[peak.lag];
    return peak;
}
}  // namespace saint
