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
              : nullptr),
      _scratch(static_cast<size_t>(samplesPerBlockPerChannel) * numChannels(channelFormat)) {}

void Preprocessor::processBlock(float* audio) {
    _lowpass.process(audio, _samplesPerBlockPerChannel);
    if (_rightLowpass) {
        _rightLowpass->process(audio, _samplesPerBlockPerChannel);
    }
}

const std::vector<float>& Preprocessor::processBlock(const float* audio) {
    const auto n = _samplesPerBlockPerChannel * _numChannels;
    std::copy(audio, audio + n, _scratch.begin());
    processBlock(_scratch.data());
    return _scratch;
}

}  // namespace saint