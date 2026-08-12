#include "PitchDetectionSmoother.h"

#include <cmath>

namespace saint {
namespace {
constexpr auto C = 0.95f;
}  // namespace

PitchDetectionSmoother::PitchDetectionSmoother(std::unique_ptr<PitchDetector> innerDetector,
                                               int blocksPerSecond)
    : _innerDetector(std::move(innerDetector)), _coef(std::pow(C, 100.0 / blocksPerSecond)) {}

PitchDetectionResult PitchDetectionSmoother::process(const float* input, DebugOutput* debugOutput,
                                                     std::vector<float>* debugOutputSignal) {
    const auto newValue = _innerDetector->process(input, debugOutput, debugOutputSignal);
    if (newValue.pitch > 0 && _lastValue == 0) {
        _lastValue = newValue.pitch;
    } else if (newValue.pitch == 0) {
        _lastValue = 0;
    }
    _lastValue = (1 - _coef) * newValue.pitch + _coef * _lastValue;
    return {_lastValue, newValue.bucket};
}

int PitchDetectionSmoother::delaySamples() const {
    return _innerDetector->delaySamples();
}

std::pair<float, float> PitchDetectionSmoother::pitchSearchRange() const {
    return _innerDetector->pitchSearchRange();
}

}  // namespace saint