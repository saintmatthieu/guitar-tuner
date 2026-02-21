#pragma once

#include "FIRFilter.h"

namespace saint {
class Upsampler {
   public:
    Upsampler(int sampleRate, int factor, int cutoffFreqHz);

    /**
     * @brief assumes mono input
     */
    std::vector<float> process(const float* input, int numSamples);

    int delaySamples() const {
        return _filter.delaySamples();
    }

   private:
    // FIR filter with linear phase response
    // Using more taps than IIR order for comparable frequency response
    // Odd number of taps ensures Type I linear phase (symmetric)
    static constexpr auto numTaps = 31;
    FIRFilter<numTaps> _filter;
    const int _factor;
};
}  // namespace saint