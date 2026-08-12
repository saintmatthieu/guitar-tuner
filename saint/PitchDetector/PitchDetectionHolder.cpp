#include "PitchDetectionHolder.h"

#include <algorithm>  // max
#include <cmath>      // round

#include "PitchDetector.h"

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

PitchDetectionResult PitchDetectionHolder::process(const float* input, DebugOutput* debugOutput,
                                                   std::vector<float>* debugOutputSignal) {
    const auto result = _innerDetector->process(input, debugOutput, debugOutputSignal);
    if (result.pitch > 0.f) {
        // Tracking: remember the pitch to hold and re-arm the hold window.
        _heldResult = result;
        _framesHeld = 0;
        return result;
    }
    if (_heldResult.pitch > 0.f && _framesHeld < _maxHoldFrames) {
        ++_framesHeld;
        if (debugOutput != nullptr) {
            (*debugOutput)["hold"] = 1.f;
        }
        return _heldResult;
    }
    return {0.f, PitchBucket::noPitch};
}

int PitchDetectionHolder::delaySamples() const {
    return _innerDetector->delaySamples();
}

}  // namespace saint
