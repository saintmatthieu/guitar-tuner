#include "PitchDetectorImpl.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>

#include "PitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
constexpr auto cutoffFreq = 1500;

constexpr float defaultLeastFrequencyToDetect = /* A0 */ 27.5f;

int getFftOrder(int windowSize) {
    return static_cast<int>(ceilf(log2f((float)windowSize)));
}

int getFftSizeSamples(int windowSize) {
    constexpr auto zeroPadding = 2;
    return 1 << (getFftOrder(windowSize) + zeroPadding);
}

float pitchToFrequency(const Pitch& pitch) {
    if (PitchClass::OneKiloHz == pitch.pitchClass) {
        return 1000 * (1 << pitch.octave);
    }
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

float disambiguateFundamentalIndex(float priorIndex, const std::vector<float>& dbSpectrum) {
    // clang-format off

    // `priorIndex` is likely a good approximation, but could also be the actual value by 2, 3 or 4 (harmonics
    // being interpreted as fundamental).
    //
    // * For each hypothesis, look for the local maxima within `k*f0 + [-B, B]`, where `k \in [1, 2, ..., N]`
    //   and `B` with the main lobe width. These must exceed -60dB or they are ignored.
    // * Get the error vector with values (fk - k * f0)², where fk is the frequency (estimated with quadratic fit) of the `k`th peak.
    // * Derive a vector of weights `w = dBk / 60 + 1`, where dBk is the value of the `k`th peak.
    // * The score is the inner product of the error and weight vectors.

    // clang-format on

    constexpr auto K = 5;
    constexpr auto D = 4;
    std::vector<float> estimates(D);
    std::vector<float> scores(D);
    for (auto d = 1; d <= D; ++d) {
        const auto f0 = priorIndex / d;
        std::vector<float> indices(K);
        std::vector<float> peakDbs(K);
        std::vector<int> harmonicNumbers(K);
        std::iota(harmonicNumbers.begin(), harmonicNumbers.end(), 1);

        for (const auto k : harmonicNumbers) {
            const auto fk = k * f0;
            const auto peakIndex =
                utils::getIndexOfClosestLocalMaximum(dbSpectrum, static_cast<int>(fk + .5f));
            const float fractionalPart = utils::quadFit(dbSpectrum.data() + peakIndex - 1);
            const float refinedPeakIndex = peakIndex + fractionalPart;
            peakDbs[k - 1] = dbSpectrum[peakIndex];
            indices[k - 1] = refinedPeakIndex;
        }

        const auto a = utils::leastSquareFit(harmonicNumbers, indices);
        std::vector<float> errors(K);
        std::transform(harmonicNumbers.begin(), harmonicNumbers.end(), errors.begin(),
                       [a, &indices](int k) {
                           const auto error = k * a - indices[k - 1];
                           return error * error;
                       });
        const auto score = std::accumulate(errors.begin(), errors.end(), 0.f);

        estimates[d - 1] = a;
        scores[d - 1] = score;
    }

    const auto it = std::min_element(scores.begin(), scores.end());
    const auto d = std::distance(scores.begin(), it) + 1;

    return estimates[d - 1];
}
}  // namespace

float PitchDetectorImpl::disambiguateEstimate(float priorEstimate,
                                              const std::vector<float>& dbSpectrum) const {
    const auto priorIndex = priorEstimate / _binFreq;
    return disambiguateFundamentalIndex(priorIndex, dbSpectrum) * _binFreq;
}

float PitchDetectorImpl::getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const {
    const auto& vec = cepstrumData.vec();

    // We're using this for a tuner, so look between 30Hz (a detuned E0 on a bass
    // guitar) and 500Hz (Ukulele high A is 440Hz)
    const auto maxPeriod = 1. / _minFreq;
    const auto minPeriod = 1. / _maxFreq;
    const auto leftmost = static_cast<int>(minPeriod * _sampleRate);
    const auto rightmost = std::min<int>(maxPeriod * _sampleRate, vec.size() / 2);
    const auto it = std::max_element(vec.begin() + leftmost, vec.begin() + rightmost);
    auto maxCepstrumIndex = std::distance(vec.begin(), it);
    if (maxCepstrumIndex == 0)
        return 0.f;
    else if (maxCepstrumIndex == vec.size())
        return static_cast<float>(_sampleRate) / maxCepstrumIndex;

    const auto bestIndex =
        takeThisIndexInstead(vec, leftmost, maxCepstrumIndex).value_or(maxCepstrumIndex);

    return static_cast<float>(_sampleRate) / bestIndex;
}

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
      _binFreq(static_cast<float>(sampleRate) / _fftSize),
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

float PitchDetectorImpl::refineEstimateBasedOnStrongestHarmonic(
    const std::vector<float>& logSpectrum, float targetFreq) const {
    constexpr auto numHarmonics = 5;
    std::vector<int> peakBins(numHarmonics);
    for (auto k = 1; k <= numHarmonics; ++k) {
        const auto targetBin = static_cast<int>(targetFreq * k / _binFreq + 0.5f);
        const auto peakBin = utils::getIndexOfClosestLocalMaximum(logSpectrum, targetBin);
        peakBins[k - 1] = peakBin;
    }
    const auto it = std::max_element(peakBins.begin(), peakBins.end(),
                                     [&](int i, int j) { return logSpectrum[i] < logSpectrum[j]; });
    const int k = std::distance(peakBins.begin(), it) + 1;
    const int peakBin{*it};  // make sure there is no narrowing conversion

    // parabolic interpolation
    if (peakBin <= 0 || peakBin >= logSpectrum.size() - 1) {
        return static_cast<float>(peakBin) * _binFreq / k;
    }

    const auto delta = utils::quadFit(&logSpectrum[peakBin - 1]);

    return (static_cast<float>(peakBin) + delta) * _binFreq / k;
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
    _logger->Log(freq.data(), freq.size(), "spectrum",
                 [](const std::complex<float>& X) { return std::abs(X); });

    std::vector<float> dbSpectrum(freq.size());
    utils::getDbSpectrum(freq, dbSpectrum.data(), freq.size());
    _logger->Log(dbSpectrum.data(), dbSpectrum.size(), "dbSpectrum");

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

    constexpr auto threshold = 0.88f;
    if (maximum < threshold) {
        return 0.f;
    }

    const auto cepstrumEstimate = getCepstrumPeakFrequency(_cepstrumData);

    const auto refinedEstimate =
        refineEstimateBasedOnStrongestHarmonic(dbSpectrum, cepstrumEstimate);

    const auto disambiguatedEstimate = disambiguateEstimate(refinedEstimate, dbSpectrum);

    return disambiguatedEstimate;
}

}  // namespace saint
