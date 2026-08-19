#pragma once

#include <memory>

#include "PitchDetector.h"
#include "PitchDetectorTypes.h"

namespace saint {
/**
 * @brief Decides what a frame the in-range detector reports nothing for really is: silence, or a
 * string sounding outside the tuning's range.
 */
class OutRangePitchDetector : public PitchDetector {
   public:
    explicit OutRangePitchDetector(std::unique_ptr<PitchDetector> inRangeDetector,
                                   OutRangeConfig = {});

    PitchDetectionResult process(const float* input, DebugOutput* = nullptr,
                                 std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;
    std::pair<float, float> pitchSearchRange() const override;

   private:
    const std::unique_ptr<PitchDetector> _inRangeDetector;
    const float _presenceThreshold;
    DebugOutput _debugOutput;
};
}  // namespace saint
