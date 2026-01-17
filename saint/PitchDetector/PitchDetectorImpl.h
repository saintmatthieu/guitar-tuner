#pragma once

#include "PitchDetector.h"

#include <pffft.hpp>
#include <ringbuffer.hpp>

#include <array>
#include <functional>
#include <optional>

class FormantShifterLoggerInterface;

namespace saint {

// PFFT memory alignment requirement
template <typename T> struct alignas(16) Aligned {
  T value;
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
  pffft::Fft<float> _fwdFft;
  pffft::Fft<float> _cepstrumFft;
  const std::vector<float> _halfWindow;
  std::array<jnk0le::Ringbuffer<float, maxBlockSize>, 2> _ringBuffers;
  std::array<float, 2> _maxima;
  int _ringBufferIndex = 0;
  const std::vector<float> _lpWindow;
  const std::vector<float> _windowXcor;
  const int _lastSearchIndex;
  std::optional<float> _detectedPitch;
};
} // namespace saint
