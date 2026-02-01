#include "PitchDetectorImpl.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>

#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
constexpr auto cutoffFreq = 1500;

constexpr float log2ToDb = 20 / 3.321928094887362f;
constexpr float defaultLeastFrequencyToDetect = /* A0 */ 27.5f;

int getFftOrder(int windowSize) {
    return static_cast<int>(ceilf(log2f((float)windowSize)));
}

int getFftSizeSamples(int windowSize) {
    constexpr auto zeroPadding = 2;
    return 1 << (getFftOrder(windowSize) + zeroPadding);
}

float pitchToFrequency(const Pitch& pitch) {
    const std::unordered_map<PitchClass, int> semitonesFromA{
        {PitchClass::C, -9},  {PitchClass::Db, -8}, {PitchClass::D, -7},  {PitchClass::Eb, -6},
        {PitchClass::E, -5},  {PitchClass::F, -4},  {PitchClass::Gb, -3}, {PitchClass::G, -2},
        {PitchClass::Ab, -1}, {PitchClass::A, 0},   {PitchClass::Bb, 1},  {PitchClass::B, 2},
    };
    const int semitonesFromA4 = semitonesFromA.at(pitch.pitchClass) + (pitch.octave - 4) * 12;
    return 440.f * std::pow(2.f, semitonesFromA4 / 12.f);
}

float getMinFreq(const std::optional<PitchDetector::Config>& config) {
    return config && config->lowestPitch.has_value() ? pitchToFrequency(*config->lowestPitch)
                                                     : defaultLeastFrequencyToDetect;
}

float getMaxFreq(const std::optional<PitchDetector::Config>& config) {
    return config && config->highestPitch.has_value() ? pitchToFrequency(*config->highestPitch)
                                                      : 2000.f;
}

int getWindowSizeSamples(int sampleRate, utils::WindowType windowType,
                         const std::optional<PitchDetector::Config>& config) {
    const auto minFreq = getMinFreq(config);
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

std::vector<float> getWindowXCorr(RealFft& fft, const std::vector<float>& window,
                                  const std::vector<float>& lpWindow) {
    Aligned<std::vector<float>> xcorrAligned;
    auto& xcorr = xcorrAligned.value;
    xcorr.resize(fft.size);
    std::copy(window.begin(), window.end(), xcorr.begin());
    std::fill(xcorr.begin() + window.size(), xcorr.end(), 0.f);
    std::vector<std::complex<float>> freq = getSpectrum(fft, xcorr.data());
    getXCorr(fft, xcorr, freq, lpWindow);
    return xcorr;
}

std::optional<int> takeThisIndexInstead(const std::vector<float>& cepstrum, int leftmost,
                                        int maxValuedIndex) {
    // Although we do "low-pass lifter" the cepstrum to reduce the risk of confusing the first
    // harmonic for the fundamental, it still happens. Liftering some introduces damage, according
    // to benchmarking.
    // Let's check if there is a peak near maxValuedIndex / 2 and if it rivals that at
    // maxValuedIndex in terms of value. If it does, let's go for this one.
    auto halfIndex = maxValuedIndex / 2;

    if (halfIndex < leftmost) {
        return {};
    }

    // climb to the top
    while (halfIndex + 1 < cepstrum.size() && cepstrum[halfIndex + 1] >= cepstrum[halfIndex]) {
        ++halfIndex;
    }
    while (halfIndex - 1 >= 0 && cepstrum[halfIndex - 1] > cepstrum[halfIndex]) {
        --halfIndex;
    }
    if (const auto tooFar = std::abs(maxValuedIndex / 2 - halfIndex) > 1) {
        return {};
    }

    const auto ratio = cepstrum[halfIndex] / cepstrum[maxValuedIndex];
    if (ratio > 0.78f) {
        return halfIndex;
    } else {
        return {};
    }
}

double getCepstrumPeakFrequency(const CepstrumData& cepstrumData, int sampleRate, float minFreq,
                                float maxFreq) {
    const auto& vec = cepstrumData.vec();

    // We're using this for a tuner, so look between 30Hz (a detuned E0 on a bass
    // guitar) and 500Hz (Ukulele high A is 440Hz)
    const auto maxPeriod = 1. / minFreq;
    const auto minPeriod = 1. / maxFreq;
    const auto leftmost = static_cast<int>(minPeriod * sampleRate);
    const auto rightmost = std::min<int>(maxPeriod * sampleRate, vec.size() / 2);
    const auto it = std::max_element(vec.begin() + leftmost, vec.begin() + rightmost);
    auto maxCepstrumIndex = std::distance(vec.begin(), it);
    if (maxCepstrumIndex == 0)
        return 0;
    else if (maxCepstrumIndex == vec.size())
        return static_cast<float>(sampleRate) / maxCepstrumIndex;

    const auto bestIndex =
        takeThisIndexInstead(vec, leftmost, maxCepstrumIndex).value_or(maxCepstrumIndex);

    return sampleRate / bestIndex;
}
}  // namespace

PitchDetectorImpl::PitchDetectorImpl(int sampleRate, ChannelFormat channelFormat,
                                     int samplesPerBlockPerChannel,
                                     const std::optional<Config>& config,
                                     std::unique_ptr<PitchDetectorLoggerInterface> logger)
    : _sampleRate(sampleRate),
      _channelFormat(channelFormat),
      _blockSize(samplesPerBlockPerChannel),
      _logger(std::move(logger)),
      _windowType(utils::WindowType::MinimumThreeTerm),
      _window(utils::getAnalysisWindow(getWindowSizeSamples(sampleRate, _windowType, config),
                                       _windowType)),
      _fftSize(getFftSizeSamples(static_cast<int>(_window.size()))),
      _fwdFft(_fftSize),
      _cepstrumData(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _minFreq(getMinFreq(config)),
      _maxFreq(getMaxFreq(config)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / _minFreq))),
      _windowXcor(getWindowXCorr(_fwdFft, _window, _lpWindow)),
      _latencySamples(std::max(static_cast<int>(_window.size()) - samplesPerBlockPerChannel, 0)),
      _audioBuffer(_latencySamples, 0.f) {
    //
    _audioBuffer.reserve(_window.size());
    _logger->SamplesRead(-_latencySamples);
}

float PitchDetectorImpl::process(const float* audio, float* presenceScore) {
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
            std::cerr << "PitchDetectorImpl::process called before enough samples were read\n";
        }
        return 0.f;
    }

    std::vector<float> time(_fftSize);
    _logger->StartNewEstimate();

    // Copy the most recent window of samples
    const auto bufferStart = _audioBuffer.end() - _window.size();
    std::copy(bufferStart, _audioBuffer.end(), time.begin());

    // Remove old samples, keeping only what's needed for the next window
    const auto samplesToKeep = _window.size() - _blockSize;
    _audioBuffer.erase(_audioBuffer.begin(), _audioBuffer.end() - samplesToKeep);
    std::fill(time.begin() + _window.size(), time.begin() + _fftSize, 0.f);

    _logger->SamplesRead(_blockSize);
    _logger->Log(_sampleRate, "sampleRate");
    _logger->Log(_fftSize, "fftSize");
    _logger->Log(_cepstrumData.fft.size, "cepstrumFftSize");
    _logger->Log(time.data(), time.size(), "inputAudio");

    // zero all samples below -60dB
    std::for_each(time.begin(), time.end(), [](float& x) {
        constexpr float threshold = 0.001f;
        if (std::abs(x) < threshold) {
            x = 0.f;
        }
    });

    applyWindow(_window, time);
    _logger->Log(time.data(), time.size(), "windowedAudio");

    // Forward FFT
    std::vector<std::complex<float>> freq = getSpectrum(_fwdFft, time.data());

    // Cepstrum analysis
    takeCepstrum(freq, _cepstrumData, *_logger);

    // Compute cross-correlation
    getXCorr(_fwdFft, time, freq, _lpWindow);
    _logger->Log(time.data(), time.size(), "xcorr");
    _logger->Log(time.data(), time.size(), "xcorrFlattened", _xcorrTransform);

    _logger->EndNewEstimate(nullptr, 0);

    auto maxIndex = 0;
    auto wentNegative = false;
    auto maximum = 0.f;
    for (auto i = 0; i < _lastSearchIndex; ++i) {
        wentNegative |= time[i] < 0;
        if (wentNegative && time[i] > maximum) {
            maximum = time[i];
            maxIndex = i;
        }
    }

    maximum /= _windowXcor[maxIndex];
    if (presenceScore) {
        *presenceScore = maximum;
    }

    constexpr auto threshold = 0.851758f;
    if (maximum < threshold) {
        return 0.f;
    }

    const auto cepstrumEstimate =
        getCepstrumPeakFrequency(_cepstrumData, _sampleRate, _minFreq, _maxFreq);

    return static_cast<float>(cepstrumEstimate);
}
}  // namespace saint
