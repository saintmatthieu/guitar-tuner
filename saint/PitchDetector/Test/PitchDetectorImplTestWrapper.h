#pragma once

#include <memory>

#include "FixedBlockPitchDetector.h"
#include "PitchDetectorImpl.h"

namespace saint {

/**
 * @brief A pitch detector without temporal filtering.
 * Wraps PitchDetectorImpl and exposes the FixedBlockPitchDetector interface.
 * Use this when testWithMedianFilter=false.
 */
class PitchDetectorImplTestWrapper : public FixedBlockPitchDetector {
   public:
    explicit PitchDetectorImplTestWrapper(std::unique_ptr<PitchDetectorImpl> impl);
    ~PitchDetectorImplTestWrapper() override = default;

    float process(const float* input, DebugOutput*, std::vector<float>*) override;
    int delaySamples() const override;

   private:
    std::unique_ptr<PitchDetectorImpl> _impl;
};

}  // namespace saint
