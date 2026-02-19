#pragma once

#include <memory>

#include "ButterworthFilter.h"

namespace saint {
class Preprocessor {
   public:
    Preprocessor(int sampleRate, int numChannels);

    void process(float* audio, int numFrames);

   private:
    static constexpr auto cutoffFreq = 5000.0;
    static constexpr auto filterOrder = 6;

    ButterworthFilter<filterOrder> _lowpass;
    const std::unique_ptr<ButterworthFilter<filterOrder>> _rightLowpass;
};
}  // namespace saint