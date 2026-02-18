#include "PitchDetectorImplTestWrapper.h"

namespace saint {

PitchDetectorImplTestWrapper::PitchDetectorImplTestWrapper(std::unique_ptr<PitchDetectorImpl> impl)
    : _impl(std::move(impl)) {}

float PitchDetectorImplTestWrapper::process(const float* input, DebugOutput* debugOutput) {
    return _impl->process(input, debugOutput);
}

int PitchDetectorImplTestWrapper::delaySamples() const {
    return _impl->delaySamples();
}

}  // namespace saint
