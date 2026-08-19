#pragma once

#include <memory>

#include "InRangePitchDetector.h"
#include "PitchDetector.h"

namespace saint {

/**
 * @brief A pitch detector without temporal filtering.
 * Wraps InRangePitchDetector and exposes the PitchDetector interface.
 * Use this when testWithMedianFilter=false.
 */
class PitchDetectorImplTestWrapper : public PitchDetector {
   public:
    explicit PitchDetectorImplTestWrapper(std::unique_ptr<InRangePitchDetector> impl);
    ~PitchDetectorImplTestWrapper() override = default;

    PitchDetectionResult process(const float* input, DebugOutput*, std::vector<float>*) override;
    int delaySamples() const override;
    std::pair<float, float> pitchSearchRange() const override;

   private:
    std::unique_ptr<InRangePitchDetector> _impl;
};

}  // namespace saint
