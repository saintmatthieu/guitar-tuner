#pragma once

#include <memory>

#include "ButterworthFilter.h"
#include "PitchDetectorTypes.h"

namespace saint {
class Preprocessor {
   public:
    Preprocessor(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel);

    void processBlock(float* audio);
    std::vector<float> processBlock(const float* audio);

   private:
    static constexpr auto cutoffFreq = 5000.0;
    static constexpr auto filterOrder = 6;

    const int _samplesPerBlockPerChannel;
    const int _numChannels;

    ButterworthFilter<filterOrder> _lowpass;
    const std::unique_ptr<ButterworthFilter<filterOrder>> _rightLowpass;
};
}  // namespace saint