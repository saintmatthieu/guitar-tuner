#include "Preprocessor.h"

namespace saint {
Preprocessor::Preprocessor(int sampleRate, ChannelFormat channelFormat,
                           int samplesPerBlockPerChannel, int decimationFactor)
    : _samplesPerBlockPerChannel(samplesPerBlockPerChannel),
      _numChannels(numChannels(channelFormat)),
      _decimationFactor(decimationFactor),
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
    _scratch.resize(static_cast<size_t>(n));
    std::copy(audio, audio + n, _scratch.begin());
    processBlock(_scratch.data());  // anti-alias low-pass, full rate, in place
    if (_decimationFactor > 1) {
        decimate();
    }
    return _scratch;
}

void Preprocessor::decimate() {
    // Keep every D-th frame, compacting in place. The output frame index never
    // overtakes the input index, so the in-place copy is safe. _decimationPhase
    // carries the keep/drop pattern across block boundaries, so the decimated rate
    // is exactly fs/D even when the block size is not a multiple of D.
    int outFrame = 0;
    for (int in = 0; in < _samplesPerBlockPerChannel; ++in) {
        if (_decimationPhase == 0) {
            for (int ch = 0; ch < _numChannels; ++ch) {
                _scratch[static_cast<size_t>(outFrame) * _numChannels + ch] =
                    _scratch[static_cast<size_t>(in) * _numChannels + ch];
            }
            ++outFrame;
        }
        _decimationPhase = (_decimationPhase + 1) % _decimationFactor;
    }
    _scratch.resize(static_cast<size_t>(outFrame) * _numChannels);
}

}  // namespace saint
