#include "PitchDetectionSmoother.h"

namespace saint {
PitchDetectionSmoother::PitchDetectionSmoother(std::unique_ptr<PitchDetector> innerDetector)
    : _innerDetector(std::move(innerDetector)) {}

float PitchDetectionSmoother::process(const float* input, DebugOutput* debugOutput,
                                      std::vector<float>* debugOutputSignal) {
    return _innerDetector->process(input, debugOutput, debugOutputSignal);
}

int PitchDetectionSmoother::delaySamples() const {
    return _innerDetector->delaySamples();
}

}  // namespace saint