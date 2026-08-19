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

std::pair<float, float> PitchDetectionHolder::pitchSearchRange() const {
    return _innerDetector->pitchSearchRange();
}

PitchDetectionResult PitchDetectionHolder::process(const float* input, DebugOutput* debugOutput,
                                                   std::vector<float>* debugOutputSignal) {
    const auto result = _innerDetector->process(input, debugOutput, debugOutputSignal);
    if (result.bucket.has_value()) {
        // Tracking: remember the pitch to hold and re-arm the hold window.
        _heldResult = result;
        _framesHeld = 0;
        return result;
    }
    if (_heldResult.bucket.has_value() && _framesHeld < _maxHoldFrames) {
        ++_framesHeld;
        if (debugOutput != nullptr) {
            (*debugOutput)["hold"] = 1.f;
        }
        return _heldResult;
    }
    return {};
}

int PitchDetectionHolder::delaySamples() const {
    return _innerDetector->delaySamples();
}

}  // namespace saint
