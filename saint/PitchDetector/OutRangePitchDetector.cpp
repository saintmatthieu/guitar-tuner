#include "OutRangePitchDetector.h"

namespace saint {
OutRangePitchDetector::OutRangePitchDetector(std::unique_ptr<PitchDetector> inRangeDetector)
    : _inRangeDetector(std::move(inRangeDetector)) {}

PitchDetectionResult OutRangePitchDetector::process(const float* input, DebugOutput* debugOutput,
                                                    std::vector<float>* debugOutputSignal) {
    return _inRangeDetector->process(input, debugOutput, debugOutputSignal);
}

int OutRangePitchDetector::delaySamples() const {
    return _inRangeDetector->delaySamples();
}

std::pair<float, float> OutRangePitchDetector::pitchSearchRange() const {
    return _inRangeDetector->pitchSearchRange();
}
}  // namespace saint
