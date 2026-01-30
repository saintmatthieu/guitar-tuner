#pragma once

#include <vector>

#include "PitchDetector.h"

namespace saint {
class PitchDetectorMedianFilter : public PitchDetector {
   public:
    PitchDetectorMedianFilter(int sampleRate, int blockSize,
                              std::unique_ptr<PitchDetector> innerDetector);

    ~PitchDetectorMedianFilter() override = default;

    float process(const float* input, float* presenceScore) override;
    float process(const float* input, float* presenceScore, float* unfilteredEstimate);

   private:
    const std::unique_ptr<PitchDetector> _innerDetector;
    std::vector<float> _buffer;
};
}  // namespace saint