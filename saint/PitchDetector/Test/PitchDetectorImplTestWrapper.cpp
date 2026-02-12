#include "PitchDetectorImplTestWrapper.h"

namespace saint {

PitchDetectorImplTestWrapper::PitchDetectorImplTestWrapper(std::unique_ptr<PitchDetectorImpl> impl)
    : _impl(std::move(impl)) {}

float PitchDetectorImplTestWrapper::process(const float* input, float* presenceScore) {
    return _impl->process(input, presenceScore);
}

int PitchDetectorImplTestWrapper::delaySamples() const {
    return _impl->delaySamples();
}

}  // namespace saint
