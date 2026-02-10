#pragma once

#include <vector>

#include "PitchDetectorTypes.h"
#include "RealFft.h"
#include "Utils.h"

namespace saint {
class PitchDetectorLoggerInterface;

class FrequencyDomainTransformer {
   public:
    FrequencyDomainTransformer(int sampleRate, ChannelFormat channelFormat,
                               int samplesPerBlockPerChannel, float minFreq,
                               PitchDetectorLoggerInterface& logger);

    std::vector<std::complex<float>> process(const float*);

    int delaySamples() const {
        return windowSizeSamples() / 2;
    }

    int windowSizeSamples() const {
        return static_cast<int>(_window.size());
    }

    int fftSize() const {
        return _fftSize;
    }

    const std::vector<float>& window() const {
        return _window;
    }

   private:
    const int _sampleRate;
    const ChannelFormat _channelFormat;
    const int _blockSize;
    PitchDetectorLoggerInterface& _logger;
    const utils::WindowType _windowType;
    const std::vector<float> _window;
    const int _fftSize;
    RealFft _fwdFft;
    std::vector<float> _audioBuffer;
    bool _bufferErrorLoggedAlready = false;
};
}  // namespace saint
