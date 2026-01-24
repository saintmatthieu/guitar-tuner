#include "Cepstrum.h"
#include "PitchDetectorLoggerInterface.h"
#include "Utils.h"

#include <algorithm>

namespace {
constexpr float FastLog2(float x) {
  static_assert(sizeof(float) == sizeof(int32_t));
  union {
    float val;
    int32_t x;
  } u = {x};
  auto log_2 = (float)(((u.x >> 23) & 255) - 128);
  u.x &= ~(255 << 23);
  u.x += 127 << 23;
  log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val - 0.65871759316667f;
  return log_2;
}
} // namespace

void saint::takeCepstrum(const std::complex<float> *spectrum, int N,
                         CepstrumData &cepstrumData,
                         PitchDetectorLoggerInterface &logger) {
  Aligned<std::vector<float>> alignedLogMag;
  auto &logMag = alignedLogMag.value;
  logMag.resize(cepstrumData.fft.size);

  // First bin is DC only.
  logMag[0] = FastLog2(spectrum[0].real() * spectrum[0].real());
  auto k = 1;
  std::transform(spectrum + 1, spectrum + cepstrumData.halfWindow.size(),
                 logMag.begin(), [&](const std::complex<float> &X) {
                   const auto power = X.real() * X.real() + X.imag() * X.imag();
                   const auto w = cepstrumData.halfWindow[k++];
                   return w * FastLog2(power);
                 });

  // No need to set the middle values to zero, `resize` already did that.

  // Now we mirror about the half
  std::reverse_copy(logMag.begin() + 1, logMag.begin() + k - 1,
                    logMag.end() - (k - 2));

  logger.Log(logMag.data(), logMag.size(), "logMagSpectrum");

  cepstrumData.fft.forward(logMag.data(), cepstrumData.ptr());

  // PFFFT wrote cepstrumData.vec.size() / 2 complex values in cepstrumData.
  // Since logMag is symmetric, the imaginary parts will be (approximately)
  // zero. We collapse the data into real values.

  // For convenience, we reinterpret the cepstrum data as complex.
  const auto complexCepstrum =
      reinterpret_cast<std::complex<float> *>(cepstrumData.ptr());
  // Now we collapse.
  for (auto i = 1; i < cepstrumData.fft.size / 2; ++i) {
    cepstrumData.vec()[i] = complexCepstrum[i].real();
  }

  logger.Log(cepstrumData.ptr(), cepstrumData.vec().size() / 2, "cepstrum");

  // Apply a gentle exponentially decaying filter to reduce the risk of
  // harmonics dominating the cepstrum peak detection.
  const float decayFactor = 0.998f;
  float decay = 1.f;
  for (auto i = 0; i < cepstrumData.fft.size / 2; ++i) {
    cepstrumData.vec()[i] *= decay;
    decay *= decayFactor;
  }

  logger.Log(cepstrumData.ptr(), cepstrumData.vec().size() / 2,
             "cepstrumFiltered");
}

namespace saint {
namespace {
std::vector<float> getHalfWindow(int fftSize) {
  // We'll only be keeping the lower part of the spectrum - the rest is just too
  // noisy. The rest will be zeroed and it will act as zero-padding.
  std::vector<float> window = utils::getAnalysisWindow(fftSize / 8);
  window.erase(window.begin(), window.begin() + window.size() / 2 - 1);
  return window;
}

} // namespace

CepstrumData::CepstrumData(int fftSize)
    : fft(RealFft(fftSize)), halfWindow(getHalfWindow(fftSize)) {
  _cepstrum.value.resize(this->fft.size);
}
} // namespace saint