#pragma once

#include <memory>
#include <vector>

#include "PitchDetector.h"
#include "PitchDetectorTypes.h"

namespace saint {
// Decorator that bridges brief presence dips in the wrapped detector's output. Once the inner
// detector has emitted a positive pitch, a momentary drop to 0 (a presence dip that would blink
// the pitch indicator off) is masked by re-emitting that last pitch for up to holdDuration
// seconds. The held value is refreshed and the window re-armed every time the inner detector
// emits a fresh positive pitch; while holding, "hold" is set to 1 in the DebugOutput (the
// TestApp reads it to colour the cursor). Kept short — it only smooths over blinks, not real
// silence; an app wanting longer persistence does so itself. A held note is dropped (0 emitted)
// once the window elapses; holdDuration == 0 disables the hold entirely.
class PitchDetectionHolder : public PitchDetector {
   public:
    PitchDetectionHolder(std::unique_ptr<PitchDetector> innerDetector, int sampleRate,
                         int blockSize, HoldConfig = {});

    float process(const float* input, DebugOutput* = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const std::unique_ptr<PitchDetector> _innerDetector;
    const int _maxHoldFrames;
    float _heldPitch = 0.f;
    int _framesHeld = 0;
};
}  // namespace saint
