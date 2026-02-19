#include "Preprocessor.h"

namespace saint {
Preprocessor::Preprocessor(int sampleRate, ChannelFormat channelFormat,
                           int samplesPerBlockPerChannel)
    : _samplesPerBlockPerChannel(samplesPerBlockPerChannel),
      _numChannels(numChannels(channelFormat)),
      _lowpass(numChannels(channelFormat), 0,
               butterworthCoefs<filterOrder>(FilterType::Lowpass, cutoffFreq, sampleRate)),
      _rightLowpass(
          channelFormat == ChannelFormat::Stereo
              ? std::make_unique<ButterworthFilter<filterOrder>>(
                    numChannels(channelFormat), 1,
                    butterworthCoefs<filterOrder>(FilterType::Lowpass, cutoffFreq, sampleRate))
              : nullptr) {}

void Preprocessor::processBlock(float* audio) {
    _lowpass.process(audio, _samplesPerBlockPerChannel);
    if (_rightLowpass) {
        _rightLowpass->process(audio, _samplesPerBlockPerChannel);
    }
}

std::vector<float> Preprocessor::processBlock(const float* audio) {
    std::vector<float> copy(_samplesPerBlockPerChannel * _numChannels);
    std::copy(audio, audio + _samplesPerBlockPerChannel * _numChannels, copy.begin());
    processBlock(copy.data());
    return copy;
}

}  // namespace saint