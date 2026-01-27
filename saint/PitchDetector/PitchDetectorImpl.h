#pragma once

#include "Cepstrum.h"
#include "PitchDetector.h"
#include "RealFft.h"

#include <ringbuffer.hpp>

#include <array>
#include <functional>
#include <optional>

class PitchDetectorLoggerInterface;

namespace saint {
class PitchDetectorImpl : public PitchDetector {
public:
  // Don't even try instantiating me if the block size exceeds this.
  PitchDetectorImpl(int sampleRate,
                    std::unique_ptr<PitchDetectorLoggerInterface> logger);
  std::optional<float> process(const float *, int,
                               float *presenceScore) override;

private:
  const float _sampleRate;
  const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
  const std::vector<float> _window;
  const int _fftSize;
  RealFft _fwdFft;
  CepstrumData _cepstrumData;
  jnk0le::Ringbuffer<float, maxBlockSize> _ringBuffer;
  const std::vector<float> _lpWindow;
  const std::vector<float> _windowXcor;
  const int _lastSearchIndex;
};
} // namespace saint
