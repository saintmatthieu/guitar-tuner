#include "FrequencyDomainTransformer.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "PitchDetectorLoggerInterface.h"
#include "PitchDetectorUtils.h"

namespace saint {
namespace {
constexpr auto cutoffFreq = 1500;

int getFftOrder(int windowSize) {
    return static_cast<int>(ceilf(log2f((float)windowSize)));
}

int getFftSizeSamples(int windowSize) {
    constexpr auto zeroPadding = 2;
    return 1 << (getFftOrder(windowSize) + zeroPadding);
}

int getWindowSizeSamples(int sampleRate, utils::WindowType windowType, float minFreq) {
    const auto numPeriods = utils::windowOrders.at(static_cast<size_t>(windowType)) * 2 + 1.3;
    const auto windowSizeMs = 1000. * numPeriods / minFreq;
    return static_cast<int>(windowSizeMs * sampleRate / 1000);
}

void applyWindow(const std::vector<float>& window, std::vector<float>& input) {
    for (auto i = 0u; i < window.size(); ++i) {
        input[i] *= window[i];
    }
}

void getXCorr(RealFft& fft, std::vector<float>& time, std::vector<std::complex<float>> freq,
              const std::vector<float>& lpWindow) {
    auto timeData = time.data();

    for (auto i = 0u; i < lpWindow.size(); ++i) {
        auto& X = freq[i];
        X *= lpWindow[i] * std::complex<float>{X.real(), -X.imag()};
    }
    std::fill(freq.data() + lpWindow.size(), freq.data() + fft.size / 2, 0.f);
    fft.inverse(freq.data(), timeData);
    if (timeData[0] < 1e-6f) {
        return;
    }
    const auto normalizer = 1.f / timeData[0];
    for (auto i = 0; i < fft.size; ++i) {
        timeData[i] *= normalizer;
    }
}

std::vector<float> getLpWindow(int sampleRate, int fftSize) {
    std::vector<float> window(fftSize / 2);
    const int cutoffBin = std::min(fftSize / 2, fftSize * cutoffFreq / sampleRate);
    const int rollOffSize = fftSize * 200 / sampleRate;
    std::fill(window.begin(), window.begin() + cutoffBin, 1.f);
    for (auto i = 0; i < rollOffSize && cutoffBin + rollOffSize < fftSize / 2; ++i) {
        window[cutoffBin + i] = 1.f - i / static_cast<float>(rollOffSize);
    }
    std::fill(window.begin() + cutoffBin + rollOffSize, window.end(), 0.f);
    return window;
}

std::vector<std::complex<float>> getSpectrum(RealFft& fft, const float* timeData) {
    Aligned<std::vector<std::complex<float>>> freqAligned;
    auto& freq = freqAligned.value;
    freq.resize(fft.size / 2);
    fft.forward(timeData, freq.data());
    return freqAligned.value;
}
}  // namespace

FrequencyDomainTransformer::FrequencyDomainTransformer(int sampleRate, ChannelFormat channelFormat,
                                                       int samplesPerBlockPerChannel, float minFreq,
                                                       PitchDetectorLoggerInterface& logger)
    : _sampleRate(sampleRate),
      _channelFormat(channelFormat),
      _blockSize(samplesPerBlockPerChannel),
      _logger(logger),
      _windowType(utils::WindowType::MinimumThreeTerm),
      _window(utils::getAnalysisWindow(getWindowSizeSamples(sampleRate, _windowType, minFreq),
                                       _windowType)),
      _fftSize(getFftSizeSamples(static_cast<int>(_window.size()))),
      _fwdFft(_fftSize),
      _audioBuffer(std::max(static_cast<int>(_window.size()) - samplesPerBlockPerChannel, 0), 0.f) {
    //
    _audioBuffer.reserve(_window.size());
    _logger.SamplesRead(-_audioBuffer.size());
}

std::vector<std::complex<float>> FrequencyDomainTransformer::process(const float* audio) {
    // Append new audio samples to buffer
    if (_channelFormat == ChannelFormat::Mono) {
        _audioBuffer.insert(_audioBuffer.end(), audio, audio + _blockSize);
    } else {
        assert(_channelFormat == ChannelFormat::Stereo);
        for (auto i = 0; i < _blockSize; ++i) {
            const auto mix = 0.5f * (audio[i * 2] + audio[i * 2 + 1]);
            _audioBuffer.push_back(mix);
        }
    }

    assert(_audioBuffer.size() >= _window.size());
    if (_audioBuffer.size() < _window.size()) {
        if (!_bufferErrorLoggedAlready) {
            _bufferErrorLoggedAlready = true;
            std::cerr
                << "FrequencyDomainTransformer::process called before enough samples were read\n";
        }
        return {};
    }

    Aligned<std::vector<float>> alignedTime;
    auto& time = alignedTime.value;
    time.resize(_fftSize);

    // Copy the most recent window of samples
    const auto bufferStart = _audioBuffer.end() - _window.size();
    std::copy(bufferStart, _audioBuffer.end(), time.begin());

    // Remove old samples, keeping only what's needed for the next window
    const auto samplesToKeep = _window.size() - _blockSize;
    _audioBuffer.erase(_audioBuffer.begin(), _audioBuffer.end() - samplesToKeep);
    std::fill(time.begin() + _window.size(), time.begin() + _fftSize, 0.f);

    _logger.SamplesRead(_blockSize);
    _logger.Log(_sampleRate, "sampleRate");
    _logger.Log(_fftSize, "fftSize");
    _logger.Log(time.data(), time.size(), "inputAudio");

    // zero all samples below -60dB
    std::for_each(time.begin(), time.end(), [](float& x) {
        constexpr float threshold = 0.001f;
        if (std::abs(x) < threshold) {
            x = 0.f;
        }
    });

    applyWindow(_window, time);
    _logger.Log(time.data(), time.size(), "windowedAudio");

    // Forward FFT
    std::vector<std::complex<float>> freq = getSpectrum(_fwdFft, time.data());
    _logger.Log(freq.data(), freq.size(), "spectrum",
                [](const std::complex<float>& X) { return std::abs(X); });

    return freq;
}
}  // namespace saint