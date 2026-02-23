#pragma once

#include <memory>

#include "PitchDetector.h"

namespace saint {
class PitchDetectionSmoother : public PitchDetector {
   public:
    PitchDetectionSmoother(std::unique_ptr<PitchDetector> innerDetector, int blocksPerSecond);

    float process(const float* input, DebugOutput* = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const std::unique_ptr<PitchDetector> _innerDetector;
    const float _coef;
    float _lastValue = 0.f;
};
}  // namespace saint