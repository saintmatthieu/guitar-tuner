#include "OnsetDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "Utils.h"

namespace saint {

namespace {
constexpr auto windowType = utils::WindowType::Hann;

int getWindowSize(int sampleRate, float minFreq) {
    const auto mainLobeWidth = utils::windowOrders.at(static_cast<size_t>(windowType)) * 2 + 1;
    const auto numPeriods = mainLobeWidth;
    const auto minPeriod = 1. / minFreq;
    return numPeriods * minPeriod * sampleRate;
}

int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}
}  // namespace

OnsetDetector::OnsetDetector(int sampleRate, ChannelFormat channelFormat,
                             int samplesPerBlockPerChannel, float minFreq, float threshold)
    : _channelFormat(channelFormat),
      _blockSize(samplesPerBlockPerChannel),
      _threshold(threshold),
      _window(utils::getAnalysisWindow<double>(getWindowSize(sampleRate, minFreq), windowType)),
      _fftSize(nextPowerOfTwo(static_cast<int>(_window.size()))),
      _fft(_fftSize),
      _audioBuffer(std::max(static_cast<int>(_window.size()) - samplesPerBlockPerChannel, 0), 0.f),
      _leastBlockCountBetweenOffsets(sampleRate / samplesPerBlockPerChannel * 0.1) {
    _audioBuffer.reserve(std::max<size_t>(_window.size(), samplesPerBlockPerChannel));
}

bool OnsetDetector::process(float* audio, DebugOutput* debugOutput) {
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
        return false;
    }

    // Window the most recent _window.size() samples into a zero-padded,
    // FFT-sized (power-of-two) buffer.
    Aligned<std::vector<float>> timeAligned;
    auto& time = timeAligned.value;
    time.assign(_fftSize, 0.f);
    const auto bufferStart = _audioBuffer.end() - _window.size();
    std::transform(bufferStart, _audioBuffer.end(), _window.begin(), time.begin(),
                   [](float x, double w) { return static_cast<float>(x * w); });

    // Remove old samples, keeping only what's needed for the next window.
    const auto samplesToKeep = std::max(static_cast<int>(_window.size()) - _blockSize, 0);
    _audioBuffer.erase(_audioBuffer.begin(), _audioBuffer.end() - samplesToKeep);

    // Forward FFT -> magnitude spectrum.
    Aligned<std::vector<std::complex<float>>> freqAligned;
    auto& freq = freqAligned.value;
    freq.resize(_fftSize / 2);
    _fft.forward(time.data(), freq.data());

    // Spectral flux: sum of the positive magnitude changes across bins. Zero on
    // the very first frame (no previous magnitudes yet).
    const auto numBins = freq.size();
    auto onsetStrength = 0.f;
    if (_prevMagnitude.size() == numBins) {
        for (size_t i = 0; i < numBins; ++i) {
            const auto mag = std::abs(freq[i]);
            const auto delta = mag - _prevMagnitude[i];
            if (delta > 0.f) {
                onsetStrength += delta;
            }
            _prevMagnitude[i] = mag;
        }
    } else {
        _prevMagnitude.resize(numBins);
        for (size_t i = 0; i < numBins; ++i) {
            _prevMagnitude[i] = std::abs(freq[i]);
        }
    }

    if (debugOutput) {
        (*debugOutput)["onsetStrength"] = onsetStrength;
    }

    const auto isOnset = onsetStrength > _threshold;

    const auto output = isOnset && _countSinceLastTrueOutput >= _leastBlockCountBetweenOffsets;
    if (output) {
        _countSinceLastTrueOutput = 0;
    } else {
        ++_countSinceLastTrueOutput;
    }

    return output;
}

bool OnsetDetector::process(const float* audio, DebugOutput* debugOutput) {
    // For const-correctness, we can copy the input to a temporary buffer and call the non-const
    // version.
    std::vector<float> copy(audio,
                            audio + _blockSize * (_channelFormat == ChannelFormat::Mono ? 1 : 2));
    return process(copy.data(), debugOutput);
}

}  // namespace saint
