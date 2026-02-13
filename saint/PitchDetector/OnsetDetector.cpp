#include "OnsetDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "Utils.h"

namespace saint {

namespace {
int getWindowSize(int sampleRate) {
    // ~23ms window — shorter than pitch detection for better temporal resolution of onsets.
    return static_cast<int>(sampleRate * 0.023);
}
}  // namespace

OnsetDetector::OnsetDetector(int sampleRate, ChannelFormat channelFormat,
                             int samplesPerBlockPerChannel)
    : _channelFormat(channelFormat),
      _blockSize(samplesPerBlockPerChannel),
      _window(utils::getAnalysisWindow(getWindowSize(sampleRate), utils::WindowType::Hann)),
      _audioBuffer(std::max(static_cast<int>(_window.size()) - samplesPerBlockPerChannel, 0), 0.f) {
    _audioBuffer.reserve(_window.size());
}

bool OnsetDetector::process(const float* audio, float* onsetStrength) {
    // Append new audio samples, converting stereo to mono if needed.
    if (_channelFormat == ChannelFormat::Mono) {
        _audioBuffer.insert(_audioBuffer.end(), audio, audio + _blockSize);
    } else {
        assert(_channelFormat == ChannelFormat::Stereo);
        for (auto i = 0; i < _blockSize; ++i) {
            const auto mix = 0.5f * (audio[i * 2] + audio[i * 2 + 1]);
            _audioBuffer.push_back(mix);
        }
    }

    if (_audioBuffer.size() < _window.size()) {
        assert(false);
        if (onsetStrength)
            *onsetStrength = 0.f;
        return false;
    }

    std::vector<float> windowed(_window.size());
    const auto bufferStart = _audioBuffer.end() - _window.size();
    std::transform(bufferStart, _audioBuffer.end(), _window.begin(), windowed.begin(),
                   [](float x, float w) { return x * w; });

    // Remove old samples, keeping only what's needed for the next window
    const auto samplesToKeep = _window.size() - _blockSize;
    _audioBuffer.erase(_audioBuffer.begin(), _audioBuffer.end() - samplesToKeep);

    const auto power = std::accumulate(audio, audio + _blockSize, 0.f,
                                       [](float acc, float val) { return acc + val * val; }) /
                       _blockSize;
    const auto novelty = power / (_prevPower + 0.0001);
    _prevPower = power;
    if (onsetStrength) {
        *onsetStrength = novelty;
    }

    // TBD
    return novelty > 100;
}

}  // namespace saint
