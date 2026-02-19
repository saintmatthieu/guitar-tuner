#pragma once

#include <memory>

#include "HighPassFilter.h"

namespace saint {
class Preprocessor {
   public:
    Preprocessor(int sampleRate, int numChannels);

    void process(float* audio, int numFrames);

   private:
    static constexpr auto cutoffFreq = 5000.0;
    static constexpr auto filterOrder = 6;

    Filter<filterOrder> _lowpass;
    const std::unique_ptr<Filter<filterOrder>> _rightLowpass;
};
}  // namespace saint