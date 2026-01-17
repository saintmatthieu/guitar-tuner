#pragma once

#include "PitchDetector.h"

#include <pffft.h>

#include <ringbuffer.hpp>

#include <array>
#include <complex>
#include <functional>
#include <optional>

class FormantShifterLoggerInterface;

namespace saint {

// PFFT memory alignment requirement
template <typename T> struct alignas(16) Aligned {
  T value;
};

class RealFft {
public:
  RealFft(int size) : setup(pffft_new_setup(size, PFFFT_REAL)), size(size) {
    work.value.resize(size);
  }

  void forward(const float *input, float *output) {
    pffft_transform_ordered(setup, input, output, work.value.data(),
                            PFFFT_FORWARD);
  }

  void forward(const float *input, std::complex<float> *output) {
    pffft_transform_ordered(setup, input, reinterpret_cast<float *>(output),
                            work.value.data(), PFFFT_FORWARD);
  }

  void inverse(const float *input, float *output) {
    pffft_transform_ordered(setup, input, output, work.value.data(),
                            PFFFT_BACKWARD);
  }

  void inverse(const std::complex<float> *input, float *output) {
    pffft_transform_ordered(setup, reinterpret_cast<const float *>(input),
                            output, work.value.data(), PFFFT_BACKWARD);
  }

  PFFFT_Setup *const setup;
  const int size;

private:
  Aligned<std::vector<float>> work;
};

class CepstrumData {
public:
  CepstrumData(RealFft fft, std::vector<float> halfWindow)
      : fft(std::move(fft)), halfWindow(std::move(halfWindow)) {
    _cepstrum.value.resize(this->fft.size);
  }

  RealFft fft;
  const std::vector<float> halfWindow;

  std::vector<float> &vec() { return _cepstrum.value; }
  float *ptr() { return _cepstrum.value.data(); }

private:
  Aligned<std::vector<float>> _cepstrum;
};

class PitchDetectorImpl : public PitchDetector {
public:
  // Don't even try instantiating me if the block size exceeds this.
  PitchDetectorImpl(int sampleRate,
                    const std::optional<float> &leastFrequencyToDetect,
                    std::unique_ptr<FormantShifterLoggerInterface> logger);
  std::optional<float> process(const float *, int) override;

private:
  const float _sampleRate;
  const std::unique_ptr<FormantShifterLoggerInterface> _logger;
  const std::vector<float> _window;
  const int _fftSize;
  RealFft _fwdFft;
  CepstrumData _cepstrumData;
  std::array<jnk0le::Ringbuffer<float, maxBlockSize>, 2> _ringBuffers;
  std::array<float, 2> _maxima;
  int _ringBufferIndex = 0;
  const std::vector<float> _lpWindow;
  const std::vector<float> _windowXcor;
  const int _lastSearchIndex;
  std::optional<float> _detectedPitch;
};
} // namespace saint
