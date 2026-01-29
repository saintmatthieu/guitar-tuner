#include "PitchDetectorImpl.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>

#include "DummyPitchDetectorLogger.h"
#include "Utils.h"

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

int getWindowSizeSamples(int sampleRate, const std::optional<PitchDetector::Config>& config) {
    // 3.3 times the fundamental period. More and that's unnecessary delay, less
    // and the detection becomes inaccurate - at least with this autocorrelation
    // method. A spectral-domain method might need less than this, since
    // autocorrelation requires there to be at least two periods within the
    // window, against 1 for a spectrum reading.
    const auto minFreq = getMinFreq(config);
    const auto windowSizeMs = 1000 * 3.3 / minFreq;
    return static_cast<int>(windowSizeMs * sampleRate / 1000);
}

void applyWindow(const std::vector<float>& window, std::vector<float>& input) {
    for (auto i = 0u; i < window.size(); ++i) {
        input[i] *= window[i];
    }
}

void getXCorr(RealFft& fft, std::vector<float>& time, const std::vector<float>& lpWindow,
              PitchDetectorLoggerInterface& logger, CepstrumData* cepstrumData = nullptr,
              std::vector<std::complex<float>>* outSpectrum = nullptr) {
    Aligned<std::vector<std::complex<float>>> freq;
    freq.value.resize(fft.size / 2);
    auto freqData = freq.value.data();
    auto timeData = time.data();

    fft.forward(timeData, freqData);

    if (outSpectrum) {
        *outSpectrum = freq.value;
    }

    if (cepstrumData) {
        takeCepstrum(freq.value.data(), freq.value.size(), *cepstrumData, logger);
    }

    for (auto i = 0; i < lpWindow.size(); ++i) {
        auto& X = freqData[i];
        X *= lpWindow[i] * std::complex<float>{X.real(), -X.imag()};
    }
    std::fill(freqData + lpWindow.size(), freqData + fft.size / 2, 0.f);
    fft.inverse(freqData, timeData);
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

std::vector<float> getWindowXCorr(RealFft& fft, const std::vector<float>& window,
                                  const std::vector<float>& lpWindow) {
    Aligned<std::vector<float>> xcorrAligned;
    auto& xcorr = xcorrAligned.value;
    xcorr.resize(fft.size);
    std::copy(window.begin(), window.end(), xcorr.begin());
    std::fill(xcorr.begin() + window.size(), xcorr.end(), 0.f);
    DummyPitchDetectorLogger logger;
    getXCorr(fft, xcorr, lpWindow, logger);
    return xcorr;
}

double getCepstrumPeakFrequency(const CepstrumData& cepstrumData, int sampleRate, float minFreq,
                                float maxFreq) {
    // We're using this for a tuner, so look between 30Hz (a detuned E0 on a bass
    // guitar) and 500Hz (Ukulele high A is 440Hz)
    const auto maxPeriod = 1. / minFreq;
    const auto minPeriod = 1. / maxFreq;
    const auto firstCepstrumSample = static_cast<int>(minPeriod * sampleRate);
    const auto& vec = cepstrumData.vec();
    const auto lastCepstrumSample = std::min<int>(maxPeriod * sampleRate, vec.size() / 2);
    const auto it =
        std::max_element(vec.begin() + firstCepstrumSample, vec.begin() + lastCepstrumSample);
    const auto maxCepstrumIndex = std::distance(vec.begin(), it);
    if (maxCepstrumIndex == 0)
        return 0;
    else if (maxCepstrumIndex == vec.size())
        return static_cast<float>(sampleRate) / maxCepstrumIndex;

    // Parabolic interpolation
    const auto prev = vec[maxCepstrumIndex - 1];
    const auto max = vec[maxCepstrumIndex];
    const auto next = vec[maxCepstrumIndex + 1];
    const auto p = 0.5f * (prev - next) / (prev + next - 2 * max);
    const auto peakIndex = maxCepstrumIndex + p;
    return sampleRate / peakIndex;
}

double getHarmonicProductSpectrumPeakFrequency(const std::vector<std::complex<float>>& spectrum,
                                               int fftSize, int sampleRate,
                                               PitchDetectorLoggerInterface& logger) {
    // Harmonic Product Spectrum: downsample and multiply harmonics
    constexpr int numHarmonics = 5;
    constexpr auto minFreq = 30.f;   // E0 on bass guitar
    constexpr auto maxFreq = 500.f;  // Ukulele high A is 440Hz

    const auto minBin = static_cast<int>(minFreq * fftSize / sampleRate);
    const auto maxBin = static_cast<int>(maxFreq * fftSize / sampleRate);

    // Compute magnitude spectrum
    std::vector<float> power(spectrum.size());
    power[0] = std::abs(spectrum[0].real());
    for (int i = 1; i < spectrum.size(); ++i) {
        power[i] = std::sqrt(spectrum[i].real() * spectrum[i].real() +
                             spectrum[i].imag() * spectrum[i].imag());
    }

    logger.Log(power.data(), power.size(), "hpsMagnitude");

    // Initialize HPS product with the fundamental spectrum
    std::vector<float> hpsProduct(maxBin + 1);
    std::fill(hpsProduct.begin(), hpsProduct.end(), 0.f);

    logger.Log(numHarmonics, "hpsNumHarmonics");
    // Multiply by downsampled harmonics
    for (int harmonic = 1; harmonic <= numHarmonics; ++harmonic) {
        std::vector<float> downsampledSpectrum(maxBin + 1);
        for (int i = 0; i <= maxBin; ++i) {
            const int harmonicBin = i * harmonic;
            if (harmonicBin < spectrum.size()) {
                downsampledSpectrum[i] = power[harmonicBin];
                hpsProduct[i] += power[harmonicBin];
            } else {
                hpsProduct[i] = 0.f;
            }
        }
        logger.Log(downsampledSpectrum.data(), downsampledSpectrum.size(),
                   ("hpsDownsampledHarmonic" + std::to_string(harmonic)).c_str());
        logger.Log(hpsProduct.data(), hpsProduct.size(),
                   ("hpsAfterHarmonic" + std::to_string(harmonic)).c_str());
    }

    // Find peak in HPS product
    const auto it = std::max_element(hpsProduct.begin() + minBin, hpsProduct.begin() + maxBin + 1);
    const auto maxBinIndex = std::distance(hpsProduct.begin(), it);

    if (maxBinIndex == 0 || maxBinIndex >= maxBin)
        return 0;

    // Parabolic interpolation for sub-bin accuracy
    const auto prev = hpsProduct[maxBinIndex - 1];
    const auto max = hpsProduct[maxBinIndex];
    const auto next = hpsProduct[maxBinIndex + 1];
    const auto p = 0.5f * (prev - next) / (prev + next - 2 * max);
    const auto peakBin = maxBinIndex + p;

    return peakBin * sampleRate / fftSize;
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
      _window(utils::getAnalysisWindow(getWindowSizeSamples(sampleRate, config),
                                       utils::WindowType::Hamming)),
      _fftSize(getFftSizeSamples(static_cast<int>(_window.size()))),
      _fwdFft(_fftSize),
      _cepstrumData(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _minFreq(getMinFreq(config)),
      _maxFreq(getMaxFreq(config)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / _minFreq))),
      _windowXcor(getWindowXCorr(_fwdFft, _window, _lpWindow)),
      _audioBuffer(std::max(static_cast<int>(_window.size()) - samplesPerBlockPerChannel, 0), 0.f) {
    //
    _audioBuffer.reserve(_window.size());
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

    // Capture spectrum for HPS analysis
    std::vector<std::complex<float>> spectrum;
    getXCorr(_fwdFft, time, _lpWindow, *_logger, &_cepstrumData, &spectrum);
    _logger->Log(time.data(), time.size(), "xcorr");

    // // Compute HPS estimate
    // const auto hpsFreq =
    //     getHarmonicProductSpectrumPeakFrequency(spectrum, _fftSize, _sampleRate, *_logger);
    // const float hpsFreqFloat = static_cast<float>(hpsFreq);
    // _logger->Log(&hpsFreqFloat, 1, "hpsFrequency");

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

    constexpr auto threshold = 0.982216f;
    if (maximum > threshold) {
        // _detectedPitch = _sampleRate / maxIndex;
        return getCepstrumPeakFrequency(_cepstrumData, _sampleRate, _minFreq, _maxFreq);
    } else {
        return 0.f;
    }
}
}  // namespace saint
