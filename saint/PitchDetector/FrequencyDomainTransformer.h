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

    // Returns a reference to an internal buffer, valid until the next call. No
    // per-call heap allocation (real-time-audio path).
    const std::vector<std::complex<float>>& process(const float*);

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
    // Reused scratch (windowed time block and its spectrum), so process()
    // allocates nothing on the audio thread.
    std::vector<float> _timeScratch;
    std::vector<std::complex<float>> _freqScratch;
};
}  // namespace saint
