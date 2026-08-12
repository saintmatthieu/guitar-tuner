#include "PitchDetectorImplTestWrapper.h"

namespace saint {

PitchDetectorImplTestWrapper::PitchDetectorImplTestWrapper(std::unique_ptr<PitchDetectorImpl> impl)
    : _impl(std::move(impl)) {}

PitchDetectionResult PitchDetectorImplTestWrapper::process(const float* input,
                                                           DebugOutput* debugOutput,
                                                           std::vector<float>* debugOutputSignal) {
    return _impl->process(input, debugOutput, debugOutputSignal);
}

int PitchDetectorImplTestWrapper::delaySamples() const {
    return _impl->delaySamples();
}

std::pair<float, float> PitchDetectorImplTestWrapper::pitchSearchRange() const {
    return _impl->pitchSearchRange();
}

}  // namespace saint
