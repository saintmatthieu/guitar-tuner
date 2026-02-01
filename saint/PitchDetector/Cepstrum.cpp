#include "Cepstrum.h"

#include <algorithm>

#include "PitchDetectorLoggerInterface.h"
#include "Utils.h"

void saint::takeCepstrum(const std::vector<std::complex<float>>& spectrum,
                         CepstrumData& cepstrumData, PitchDetectorLoggerInterface& logger) {
    Aligned<std::vector<float>> alignedLogMag;
    auto& logMag = alignedLogMag.value;
    logMag.resize(cepstrumData.fft.size);

    const auto& hw = cepstrumData.halfWindow;
    // First bin is DC only.
    logMag[0] = hw[0] * utils::FastLog2(spectrum[0].real() * spectrum[0].real());
    auto k = 1;
    std::transform(spectrum.data() + 1, spectrum.data() + cepstrumData.halfWindow.size(),
                   logMag.begin(), [&](const std::complex<float>& X) {
                       const auto power = X.real() * X.real() + X.imag() * X.imag();
                       const auto w = hw[k++];
                       return w * utils::FastLog2(power);
                   });

    // No need to set the middle values to zero, `resize` already did that.

    // Now we mirror about the half
    std::reverse_copy(logMag.begin() + 1, logMag.begin() + k - 1, logMag.end() - (k - 2));

    logger.Log(logMag.data(), logMag.size(), "logMagSpectrum");

    cepstrumData.fft.forward(logMag.data(), cepstrumData.ptr());

    // PFFFT wrote cepstrumData.vec.size() / 2 complex values in cepstrumData.
    // Since logMag is symmetric, the imaginary parts will be (approximately)
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