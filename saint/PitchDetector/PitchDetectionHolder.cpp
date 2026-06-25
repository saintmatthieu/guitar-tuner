#include "PitchDetectionHolder.h"

#include <algorithm>  // max
#include <cmath>      // round

namespace saint {
namespace {
int durationToBlocks(int sampleRate, int blockSize, float duration) {
    const auto blockDuration = static_cast<float>(blockSize) / static_cast<float>(sampleRate);
    return std::max(0, static_cast<int>(std::round(duration / blockDuration)));
}
}  // namespace

PitchDetectionHolder::PitchDetectionHolder(std::unique_ptr<PitchDetector> innerDetector,
                                           int sampleRate, int blockSize, HoldConfig config)
    : _innerDetector(std::move(innerDetector)),
      _maxHoldFrames(durationToBlocks(sampleRate, blockSize, config.holdDuration)) {}

float PitchDetectionHolder::process(const float* input, DebugOutput* debugOutput,
                                    std::vector<float>* debugOutputSignal) {
    const auto pitch = _innerDetector->process(input, debugOutput, debugOutputSignal);
    if (pitch > 0.f) {
        // Tracking: remember the pitch to hold and re-arm the hold window.
        _heldPitch = pitch;
        _framesHeld = 0;
        return pitch;
    }
    if (_heldPitch > 0.f && _framesHeld < _maxHoldFrames) {
        ++_framesHeld;
        if (debugOutput != nullptr) {
            (*debugOutput)["hold"] = 1.f;
        }
        return _heldPitch;
    }
    return 0.f;
}

int PitchDetectionHolder::delaySamples() const {
    return _innerDetector->delaySamples();
}

}  // namespace saint
