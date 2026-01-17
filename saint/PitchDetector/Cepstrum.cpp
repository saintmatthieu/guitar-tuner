#include "Cepstrum.h"
#include "FormantShifterLoggerInterface.h"
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
                         FormantShifterLoggerInterface &logger) {
  Aligned<std::vector<float>> logMag;
  // The information we're interested is doesn't exceed 3kHz. Assuming 44.1k,
  // it means we can divide the fft size by about 16. But we will mirror it to
  // enhance the periodicity, so only by 8.
  const int copiedBins = cepstrumData.halfWindow.size();

  logMag.value.resize(cepstrumData.fft.size);

  // First bin is DC only.
  logMag.value[0] = FastLog2(spectrum[0].real() * spectrum[0].real());
  auto k = 1;
  std::transform(spectrum + 1, spectrum + copiedBins, logMag.value.begin(),
                 [&](const std::complex<float> &X) {
                   const auto power = X.real() * X.real() + X.imag() * X.imag();
                   const auto w = cepstrumData.halfWindow[k++];
                   return w * FastLog2(power);
                 });

  // Mirror about the half
  std::reverse_copy(logMag.value.begin() + 1,
                    logMag.value.begin() + copiedBins - 1,
                    logMag.value.begin() + copiedBins);

  logger.Log(logMag.value.data(), logMag.value.size(), "logMagSpectrum");

  cepstrumData.fft.forward(logMag.value.data(), cepstrumData.ptr());

  // PFFFT wrote cepstrumData.vec.size() / 2 complex values in cepstrumData.
  // Since logMag is symmetric, the imaginary parts will be (approximately)
  // zero. We collapse the data into real values.

  // For convenience, we reinterpret the cepstrum data as complex.
  const auto complexCepstrum =
      reinterpret_cast<std::complex<float> *>(cepstrumData.ptr());
  // Now we collapse.
  for (auto i = 1; i < cepstrumData.fft.size; ++i) {
    cepstrumData.vec()[i] = complexCepstrum[i].real();
  }

  logger.Log(cepstrumData.ptr(), cepstrumData.vec().size() / 2, "cepstrum");
}

namespace saint {
namespace {
constexpr auto getCepstrumSize(int fftSize) {
  return fftSize / cepstrumDecimationFactor;
}

constexpr auto getCopiedSize(int fftSize) {
  return fftSize / cepstrumDecimationFactor / 2 + 1;
}

std::vector<float> getHalfWindow(int fftSize) {
  std::vector<float> window =
      utils::getAnalysisWindow(getCepstrumSize(fftSize));
  const auto copiedSize = getCopiedSize(fftSize);
  window.erase(window.begin(), window.begin() + window.size() - copiedSize);
  return window;
}

} // namespace

CepstrumData::CepstrumData(int fftSize)
    : fft(RealFft(fftSize / cepstrumDecimationFactor)),
      halfWindow(getHalfWindow(fftSize)) {
  _cepstrum.value.resize(this->fft.size);
}
} // namespace saint