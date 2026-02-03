#include "Cepstrum.h"

#include <algorithm>

#include "PitchDetectorLoggerInterface.h"
#include "Utils.h"

void saint::takeCepstrum(const std::vector<std::complex<float>>& spectrum,
                         CepstrumData& cepstrumData, PitchDetectorLoggerInterface& logger) {
    Aligned<std::vector<float>> windowedDbSpecAligned;
    auto& windowedDbSpec = windowedDbSpecAligned.value;
    windowedDbSpec.resize(cepstrumData.fft.size);

    const auto halfWindowSize = cepstrumData.halfWindow.size();

    utils::getDbSpectrum(spectrum, windowedDbSpec.data(), halfWindowSize);

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

    cepstrumData.fft.forward(windowedDbSpec.data(), cepstrumData.ptr());

    // PFFFT wrote cepstrumData.vec.size() / 2 complex values in cepstrumData.
    // Since windowedDbSpec is symmetric, the imaginary parts will be (approximately)
    // zero. We collapse the data into real values.

    // For convenience, we reinterpret the cepstrum data as complex.
    const auto complexCepstrum = reinterpret_cast<std::complex<float>*>(cepstrumData.ptr());
    // Now we collapse.
    for (auto i = 1; i < cepstrumData.fft.size / 2; ++i) {
        cepstrumData.vec()[i] = complexCepstrum[i].real();
    }

    logger.Log(cepstrumData.ptr(), cepstrumData.vec().size() / 2, "cepstrum");

    // Apply a gentle exponentially decaying filter to reduce the risk of
    // harmonics dominating the cepstrum peak detection.
    const float decayFactor = 0.9975f;
    float decay = 1.f;
    for (auto i = 0; i < cepstrumData.fft.size / 2; ++i) {
        cepstrumData.vec()[i] *= decay;
        decay *= decayFactor;
    }

    logger.Log(cepstrumData.ptr(), cepstrumData.vec().size() / 2, "cepstrumFiltered");
}

namespace saint {
namespace {
std::vector<float> getHalfWindow(int fftSize) {
    // We'll only be keeping the lower part of the spectrum - the rest is just too
    // noisy. The rest will be zeroed and it will act as zero-padding.
    std::vector<float> window = utils::getAnalysisWindow(fftSize / 8, utils::WindowType::Hann);
    window.erase(window.begin(), window.begin() + window.size() / 2 - 1);
    return window;
}

}  // namespace

CepstrumData::CepstrumData(int fftSize)
    : fft(RealFft(fftSize)), halfWindow(getHalfWindow(fftSize)) {
    _cepstrum.value.resize(this->fft.size);
}
}  // namespace saint