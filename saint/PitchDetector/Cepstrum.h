#pragma once

#include "RealFft.h"

namespace saint {
class FormantShifterLoggerInterface;

class CepstrumData {
public:
  CepstrumData(int fftSize);

  RealFft fft;
  const std::vector<float> halfWindow;

  std::vector<float> &vec() { return _cepstrum.value; }
  const std::vector<float> &vec() const { return _cepstrum.value; }
  float *ptr() { return _cepstrum.value.data(); }

private:
  Aligned<std::vector<float>> _cepstrum;
};

/**
 * @param spectrum N/2 complex values, the first of which is DC + Nyquist
 */
void takeCepstrum(const std::complex<float> *spectrum, int N,
                  CepstrumData &cepstrumData,
                  FormantShifterLoggerInterface &logger);
} // namespace saint