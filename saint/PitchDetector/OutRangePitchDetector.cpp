#include "OutRangePitchDetector.h"

namespace saint {
OutRangePitchDetector::OutRangePitchDetector(std::unique_ptr<InRangePitchDetector> inRangeDetector,
                                             OutRangeConfig config)
    : _inRangeDetector(std::move(inRangeDetector)), _presenceThreshold(config.presenceThreshold) {}

PitchDetectionResult OutRangePitchDetector::process(const float* input, DebugOutput* debugOutput,
                                                    std::vector<float>* debugOutputSignal) {
    if (debugOutput == nullptr) {
        debugOutput = &_debugOutput;
    }
    const auto inRange = _inRangeDetector->process(input, debugOutput, debugOutputSignal);
    if (inRange.bucket.has_value()) {
        return inRange;
    }

    // Nothing in range, so the question is only whether something is sounding at all - a much
    // easier one than which pitch, and the presence score already answers it. Which side of the
    // range it fell on is not decided yet; below is the case that brought us here.
    const auto presenceIt = debugOutput->find("presenceScore");
    const auto presence = presenceIt != debugOutput->end() ? presenceIt->second : 0.f;
    if (presence < _presenceThreshold) {
        return {};
    }
    return {0.f, PitchBucket::belowRange};
}

int OutRangePitchDetector::delaySamples() const {
    return _inRangeDetector->delaySamples();
}

std::pair<float, float> OutRangePitchDetector::pitchSearchRange() const {
    return _inRangeDetector->pitchSearchRange();
}

void OutRangePitchDetector::setEstimateConstraint(float constraint) {
    _inRangeDetector->setEstimateConstraint(constraint);
}
}  // namespace saint
