#pragma once

#include "Cepstrum.h"
#include "PitchDetector.h"
#include "RealFft.h"

#include <ringbuffer.hpp>

#include <array>
#include <functional>
#include <optional>

class FormantShifterLoggerInterface;

namespace saint {
class PitchDetectorImpl : public PitchDetector {
public:
  // Don't even try instantiating me if the block size exceeds this.
  PitchDetectorImpl(int sampleRate,
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
};
} // namespace saint
