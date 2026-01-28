#pragma once

#include <optional>
#include <ringbuffer.hpp>

#include "Cepstrum.h"
#include "PitchDetector.h"
#include "RealFft.h"

class PitchDetectorLoggerInterface;

namespace saint {
class PitchDetectorImpl : public PitchDetector {
   public:
    PitchDetectorImpl(int sampleRate, int blockSize, const std::optional<Config>& config,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);
    float process(const float*, float* presenceScore) override;

    int windowSizeSamples() {
        return static_cast<int>(_window.size());
    }

   private:
    const float _sampleRate;
    const int _blockSize;
    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    const std::vector<float> _window;
    const int _fftSize;
    RealFft _fwdFft;
    CepstrumData _cepstrumData;
    jnk0le::Ringbuffer<float, maxBlockSize> _ringBuffer;
    const std::vector<float> _lpWindow;
    const std::vector<float> _windowXcor;
    const float _minFreq;
    const float _maxFreq;
    const int _lastSearchIndex;
    bool _bufferErrorLoggedAlready = false;
};
}  // namespace saint
