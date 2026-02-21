#pragma once

#include <memory>

#include "PitchDetector.h"

namespace saint {
class PitchDetectionSmoother : public PitchDetector {
   public:
    PitchDetectionSmoother(std::unique_ptr<PitchDetector> innerDetector);

    float process(const float* input, DebugOutput* = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const std::unique_ptr<PitchDetector> _innerDetector;
};
}  // namespace saint