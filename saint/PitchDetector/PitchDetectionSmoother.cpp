#include "PitchDetectionSmoother.h"

#include <cmath>

namespace saint {
namespace {
constexpr auto C = 0.9f;
}  // namespace

PitchDetectionSmoother::PitchDetectionSmoother(std::unique_ptr<PitchDetector> innerDetector,
                                               int blocksPerSecond)
    : _innerDetector(std::move(innerDetector)), _coef(std::pow(C, 100.0 / blocksPerSecond)) {}

float PitchDetectionSmoother::process(const float* input, DebugOutput* debugOutput,
                                      std::vector<float>* debugOutputSignal) {
    const auto newValue = _innerDetector->process(input, debugOutput, debugOutputSignal);
    if (newValue > 0 && _lastValue == 0) {
        _lastValue = newValue;
    } else if (newValue == 0) {
        _lastValue = 0;
    }
    return _lastValue = (1 - _coef) * newValue + _coef * _lastValue;
}

int PitchDetectionSmoother::delaySamples() const {
    return _innerDetector->delaySamples();
}

}  // namespace saint