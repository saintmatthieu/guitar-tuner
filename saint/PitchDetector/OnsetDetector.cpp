#include "OnsetDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "PitchDetectorTypes.h"
#include "Utils.h"

namespace saint {

namespace {
constexpr auto windowType = WindowType::Hann;

int getWindowSize(int sampleRate, float minFreq) {
    const auto numPeriods = utils::mainLobeWidth<windowType>();
    const auto minPeriod = 1. / minFreq;
    return numPeriods * minPeriod * sampleRate;
}
}  // namespace

OnsetDetector::OnsetDetector(int sampleRate, ChannelFormat channelFormat,
                             int samplesPerBlockPerChannel, float minFreq)
    : _channelFormat(channelFormat),
      _blockSize(samplesPerBlockPerChannel),
      _window(utils::getAnalysisWindow<double>(getWindowSize(sampleRate, minFreq), windowType)),
      _audioBuffer(std::max(static_cast<int>(_window.size()) - samplesPerBlockPerChannel, 0), 0.f),
      _avgFilterLength(1. * sampleRate / samplesPerBlockPerChannel * 0.25),
      _pastPowers(_avgFilterLength, 0.f),
      _avgWindow(utils::getAnalysisWindow<double>(_avgFilterLength, WindowType::Hann)),
      _alpha(0.7 * sampleRate / samplesPerBlockPerChannel / 100),
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

    const auto bufferStart = _audioBuffer.end() - _window.size();
    std::vector<double> windowed(_window.size());
    std::transform(bufferStart, _audioBuffer.end(), _window.begin(), windowed.begin(),
                   [](double x, double w) { return x * w; });

    // Remove old samples, keeping only what's needed for the next window
    const auto samplesToKeep = std::max(static_cast<int>(_window.size()) - _blockSize, 0);
    _audioBuffer.erase(_audioBuffer.begin(), _audioBuffer.end() - samplesToKeep);

    const auto power = std::accumulate(windowed.begin(), windowed.end(), 0.f,
                                       [](float acc, float val) { return acc + val * val; });

    const auto onsetStrength = std::max(_prevPower.has_value() ? power - *_prevPower : 0., 0.);

    _prevPower = power * (1 - _alpha) + _prevPower.value_or(power) * _alpha;

    if (debugOutput) {
        (*debugOutput)["power"] = power;
        (*debugOutput)["powerAvg"] = _prevPower.value_or(0.);
        (*debugOutput)["onsetStrength"] = onsetStrength;
    }

    // To get this, we run OnsetDetectorCalibrationTests and then showOnsetDetectionHistograms.py.
    // It is set so that there are no false negatives.
    const auto isOnset = onsetStrength > 1.75109e-07;

    const auto output = isOnset && _countSinceLastTrueOutput >= _leastBlockCountBetweenOffsets;
    if (output) {
        _countSinceLastTrueOutput = 0;
    } else {
        ++_countSinceLastTrueOutput;
    }

    return output;
}

double OnsetDetector::updatePowerAverage(double newPower) {
    _pastPowers.erase(_pastPowers.begin());
    _pastPowers.push_back(newPower);

    auto avg = 0.;
    for (auto i = 0; i < _avgFilterLength; ++i) {
        avg += _pastPowers[i] * _avgWindow[i];
    }
    return avg;
}

bool OnsetDetector::process(const float* audio, DebugOutput* debugOutput) {
    // For const-correctness, we can copy the input to a temporary buffer and call the non-const
    // version.
    std::vector<float> copy(audio,
                            audio + _blockSize * (_channelFormat == ChannelFormat::Mono ? 1 : 2));
    return process(copy.data(), debugOutput);
}

}  // namespace saint
