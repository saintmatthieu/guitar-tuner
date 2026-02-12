#pragma once

#include <memory>

#include "PitchDetector.h"
#include "PitchDetectorImpl.h"

namespace saint {

/**
 * @brief A pitch detector without temporal filtering.
 * Wraps PitchDetectorImpl and exposes the PitchDetector interface.
 * Use this when testWithMedianFilter=false.
 */
class PitchDetectorImplTestWrapper : public PitchDetector {
   public:
    explicit PitchDetectorImplTestWrapper(std::unique_ptr<PitchDetectorImpl> impl);
    ~PitchDetectorImplTestWrapper() override = default;

    float process(const float* input, float* presenceScore) override;
    int delaySamples() const override;

   private:
    std::unique_ptr<PitchDetectorImpl> _impl;
};

}  // namespace saint
