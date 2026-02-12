#pragma once

#include <memory>
#include <vector>

#include "PitchDetector.h"
#include "PitchDetectorImpl.h"

namespace saint {
class PitchDetectorMedianFilter : public PitchDetector {
   public:
    PitchDetectorMedianFilter(int sampleRate, int blockSize,
                              std::unique_ptr<PitchDetectorImpl> impl);

    ~PitchDetectorMedianFilter() override = default;

    float process(const float* input, float* presenceScore) override;
    float process(const float* input, float* presenceScore, float* unfilteredEstimate);
    int delaySamples() const override;

   private:
    const int _blockSize = 0;
    const std::unique_ptr<PitchDetectorImpl> _impl;
    std::vector<float> _buffer;
    std::vector<float> _delayedScores;
};
}  // namespace saint