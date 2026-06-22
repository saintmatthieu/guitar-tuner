#pragma once

#include <memory>
#include <vector>

#include "ButterworthFilter.h"
#include "PitchDetectorTypes.h"

namespace saint {
class Preprocessor {
   public:
    Preprocessor(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel);

    void processBlock(float* audio);
    // Returns a reference to an internal buffer, valid until the next call. No
    // per-call heap allocation (real-time-audio path).
    const std::vector<float>& processBlock(const float* audio);

   private:
    static constexpr auto cutoffFreq = 5000.0;
    static constexpr auto filterOrder = 6;

    const int _samplesPerBlockPerChannel;
    const int _numChannels;

    ButterworthFilter<filterOrder> _lowpass;
    const std::unique_ptr<ButterworthFilter<filterOrder>> _rightLowpass;
    std::vector<float> _scratch;  // reused output buffer
};
}  // namespace saint