#pragma once

#include <vector>

#include "PitchDetectorTypes.h"

namespace saint {

class OnsetDetector {
   public:
    OnsetDetector(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel);

    bool process(const float* audio, float* onsetStrength = nullptr);

   private:
    const ChannelFormat _channelFormat;
    const int _blockSize;
    const std::vector<float> _window;
    std::vector<float> _audioBuffer;
    float _prevPower = 0.f;
};

}  // namespace saint
