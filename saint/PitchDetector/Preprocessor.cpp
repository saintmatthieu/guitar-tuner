#include "Preprocessor.h"

namespace saint {
Preprocessor::Preprocessor(int sampleRate, int numChannels)
    : _lowpass(numChannels, 0,
               butterworthCoefs<filterOrder>(FilterType::Lowpass, cutoffFreq, sampleRate)),
      _rightLowpass(numChannels == 2 ? std::make_unique<Filter<filterOrder>>(
                                           numChannels, 1,
                                           butterworthCoefs<filterOrder>(FilterType::Lowpass,
                                                                         cutoffFreq, sampleRate))
                                     : nullptr) {}

void Preprocessor::process(float* audio, int numFrames) {
    _lowpass.process(audio, numFrames);
    if (_rightLowpass) {
        _rightLowpass->process(audio, numFrames);
    }
}

}  // namespace saint