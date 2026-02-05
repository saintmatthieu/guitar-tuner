#include "Cepstrum.h"

#include <algorithm>

#include "PitchDetectorLoggerInterface.h"
#include "Utils.h"

void saint::toCepstrum(const std::vector<std::complex<float>>& spectrum, CepstrumData& cepstrumData,
                       PitchDetectorLoggerInterface& logger) {
    Aligned<std::vector<float>> windowedDbSpecAligned;
    auto& windowedDbSpec = windowedDbSpecAligned.value;
    windowedDbSpec.resize(cepstrumData.fft.size);

    const auto halfWindowSize = cepstrumData.halfWindow.size();

    utils::getDbSpectrum(spectrum, windowedDbSpec, halfWindowSize);

    // Apply half-windowing to reduce spectral leakage in the cepstrum.
    std::transform(windowedDbSpec.begin(), windowedDbSpec.begin() + halfWindowSize,
                   cepstrumData.halfWindow.begin(), windowedDbSpec.begin(),
                   [](float x, float w) { return x * w; });

    // Fill the rest with zeros
    const auto k = cepstrumData.fft.size / 2 - halfWindowSize + 1;
    std::fill(windowedDbSpec.begin() + halfWindowSize, windowedDbSpec.end(), 0.f);

    // Now we mirror about the half
    std::reverse_copy(windowedDbSpec.begin() + 1, windowedDbSpec.begin() + k - 1,
                      windowedDbSpec.end() - (k - 2));

    logger.Log(windowedDbSpec.data(), windowedDbSpec.size(), "windowedDbSpec");

    toCepstrum(windowedDbSpec, cepstrumData.fft, cepstrumData.aligned());

    logger.Log(cepstrumData.ptr(), cepstrumData.vec().size(), "cepstrum");
}

void saint::toCepstrum(const std::vector<float>& logSpectrum, RealFft& fft,
                       Aligned<std::vector<float>>& cepstrumAligned) {
    auto& cepstrum = cepstrumAligned.value;
    cepstrum.resize(fft.size);

    fft.forward(logSpectrum.data(), cepstrum.data());

    // PFFFT wrote cepstrumData.vec.size() / 2 complex values in cepstrumData.
    // Since windowedDbSpec is symmetric, the imaginary parts will be (approximately)
    // zero. We collapse the data into real values.

    // For convenience, we reinterpret the cepstrum data as complex.
    const auto complexCepstrum = reinterpret_cast<std::complex<float>*>(cepstrum.data());
    // Now we collapse.
    for (auto i = 1; i < fft.size / 2 + 1; ++i) {
        cepstrum[i] = complexCepstrum[i].real();
    }
    // Now mirror the rest.
    std::reverse_copy(cepstrum.begin() + 1, cepstrum.begin() + fft.size / 2,
                      cepstrum.end() - (fft.size / 2 - 1));
}

std::vector<float> saint::fromCepstrum(RealFft& fft, const float* cepstrumPtr) {
    Aligned<std::vector<float>> spectrumAligned;
    auto& spectrum = spectrumAligned.value;
    spectrum.resize(fft.size);
    fft.forward(cepstrumPtr, spectrum.data());
    const auto scale = 1.f / fft.size;
    // Because the cepstrum is symmetric, the imaginary parts are zero.
    for (auto i = 0; i < fft.size / 2 + 1; ++i) {
        spectrum[i] = spectrum[2 * i] * scale;
    }
    spectrum.resize(fft.size / 2);
    return spectrum;
}

namespace saint {
namespace {
std::vector<float> getHalfWindow(int fftSize, int sampleRate) {
    // We'll only be keeping the lower part of the spectrum - the rest is just too
    // noisy. The rest will be zeroed and it will act as zero-padding.
    constexpr float cutoffFreq = 5000.f;
    const auto cutoffBin = std::min<int>(fftSize / 2, fftSize * cutoffFreq / sampleRate);
    std::vector<float> window = utils::getAnalysisWindow(cutoffBin * 2, utils::WindowType::Hann);
    window.erase(window.begin(), window.begin() + window.size() / 2 - 1);
    return window;
}

}  // namespace

CepstrumData::CepstrumData(int fftSize, int sampleRate)
    : fft(RealFft(fftSize)), halfWindow(getHalfWindow(fftSize, sampleRate)) {
    _cepstrum.value.resize(this->fft.size);
}
}  // namespace saint