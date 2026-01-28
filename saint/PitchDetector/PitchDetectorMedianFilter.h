#pragma once

#include <vector>

#include "PitchDetector.h"

namespace saint {
class PitchDetectorMedianFilter : public PitchDetector {
   public:
    PitchDetectorMedianFilter(int sampleRate, int blockSize,
                              std::unique_ptr<PitchDetector> innerDetector);

    float process(const float* input, float* presenceScore = nullptr) override;
    ~PitchDetectorMedianFilter() override = default;

   private:
    const std::unique_ptr<PitchDetector> _innerDetector;
    std::vector<float> _buffer;
};
}  // namespace saint