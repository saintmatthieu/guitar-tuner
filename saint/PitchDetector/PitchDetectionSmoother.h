#pragma once

#include <memory>

#include "FixedBlockPitchDetector.h"

namespace saint {
class PitchDetectionSmoother : public FixedBlockPitchDetector {
   public:
    PitchDetectionSmoother(std::unique_ptr<FixedBlockPitchDetector> innerDetector,
                           int blocksPerSecond);

    float process(const float* input, DebugOutput* = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const std::unique_ptr<FixedBlockPitchDetector> _innerDetector;
    const float _coef;
    float _lastValue = 0.f;
};
}  // namespace saint