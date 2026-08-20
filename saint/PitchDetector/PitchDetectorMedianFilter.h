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
    struct InnerResult {
        float pitch = 0.f;
        int bucket = -1;
    };
    const int _blockSize = 0;
    const std::unique_ptr<OutRangePitchDetector> _inner;
    DebugOutput _debugOutput;
    const int _minInRangeCount;
    std::vector<InnerResult> _buffer;
    // Reused, so the per-block vote allocates nothing on the audio thread.
    std::vector<InnerResult> _sortScratch;
    std::vector<float> _delayedScores;
    bool _allGoodOnce = false;
};
}  // namespace saint