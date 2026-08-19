#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "OutRangePitchDetector.h"

namespace saint {
class PitchDetectorMedianFilter : public PitchDetector {
   public:
    PitchDetectorMedianFilter(int sampleRate, int blockSize,
                              std::unique_ptr<OutRangePitchDetector> inner,
                              MedianFilterConfig = {});

    ~PitchDetectorMedianFilter() override = default;

    PitchDetectionResult process(const float* input, DebugOutput*, std::vector<float>*) override;
    int delaySamples() const override;
    std::pair<float, float> pitchSearchRange() const override;

   private:
    const int _blockSize = 0;
    const std::unique_ptr<OutRangePitchDetector> _inner;
    DebugOutput _debugOutput;
    std::vector<PitchDetectionResult> _buffer;
    std::vector<float> _delayedScores;
    // Bucket of the last pitched raw result, to notice a switch between an in-range reading and
    // a below-range one (see process).
    std::optional<PitchBucket> _lastBucket;
    bool _allGoodOnce = false;
};
}  // namespace saint