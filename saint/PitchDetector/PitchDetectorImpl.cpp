#include "PitchDetectorImpl.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>

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

std::vector<std::pair<int, int>> primeCombinations(int K) {
    // If K == 4, returns {(1,2), (1,3), (1,4), (2,3), (3,4)}, i.e., all tuples with elements in [1,
    // K] whose GCD is 1.
    std::vector<std::pair<int, int>> combinations;
    for (auto i = 1; i <= K; ++i) {
        for (auto j = i + 1; j <= K; ++j) {
            if (std::gcd(i, j) == 1) {
                combinations.emplace_back(i, j);
            }
        }
    }
    return combinations;
}

struct PeakData {
    std::vector<int> indices;
    std::vector<float> values;
};

PeakData getPeaks(const std::vector<float>& spectrum, int minIndex, int maxIndex) {
    PeakData peaks;

    for (int i = minIndex; i < maxIndex - 1; ++i) {
        if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.f) {
            peaks.indices.push_back(i);
            peaks.values.push_back(spectrum[i]);
        }
    }

    if (peaks.indices.size() == 1) {
        return peaks;
    }

    // Remove peaks that aren't looking good because of interference with noise or another peak
    // that's too close.
    constexpr auto minDiffDb = 10.f;
    std::vector<int> peakIndexIndicesToRemove;
    for (size_t i = 0; i < peaks.indices.size(); ++i) {
        auto leftTroughIndex = peaks.indices[i];
        while (leftTroughIndex > 0 && spectrum[leftTroughIndex - 1] < spectrum[leftTroughIndex]) {
            --leftTroughIndex;
        }
        if (leftTroughIndex == 0 ||
            spectrum[peaks.indices[i]] - spectrum[leftTroughIndex] < minDiffDb) {
            peakIndexIndicesToRemove.push_back(i);
            continue;
        }

        auto rightTroughIndex = peaks.indices[i];
        while (rightTroughIndex + 1 < maxIndex &&
               spectrum[rightTroughIndex + 1] < spectrum[rightTroughIndex]) {
            ++rightTroughIndex;
        }
        if (rightTroughIndex + 1 == maxIndex ||
            spectrum[peaks.indices[i]] - spectrum[rightTroughIndex] < minDiffDb) {
            peakIndexIndicesToRemove.push_back(i);
        }
    }
    for (auto it = peakIndexIndicesToRemove.rbegin(); it != peakIndexIndicesToRemove.rend(); ++it) {
        peaks.indices.erase(peaks.indices.begin() + *it);
        peaks.values.erase(peaks.values.begin() + *it);
    }

    return peaks;
}

struct LineFitResult {
    float slope = 0.f;  // a
    std::vector<float> absErrors;
    float meanSquaredError = std::numeric_limits<float>::max();  // weighted sum of squared errors
};

LineFitResult leastSquareFit(const std::vector<int>& k, const PeakData& peaks,
                             const std::vector<float>& weights) {
    // Fit a line y = a*x + b to the data points (k[i], peakIndices[i]) for i in activeIndices
    // using weighted least squares.
    const auto n = k.size();
    if (n < 2) {
        return {};
    }

    std::vector<float> x(n), y(n), w(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(k[i]);
        y[i] = static_cast<float>(peaks.indices[i]);
        w[i] = weights[i];
    }

    const auto a = utils::leastSquareFit(x, y, w);

    // Compute weighted sum of squared errors
    float meanSquaredError = 0.f;
    std::vector<float> absErrors(n);
    for (size_t i = 0; i < n; ++i) {
        const float residual = a * x[i] - y[i];
        absErrors[i] = std::abs(residual);
        meanSquaredError += w[i] * residual * residual;
    }
    meanSquaredError /= n;

    return {a, std::move(absErrors), meanSquaredError};
}

template <typename IntContainer>
constexpr int getGcd(const IntContainer& ints) {
    if (ints.empty()) {
        return 0;
    }
    auto result = *ints.begin();
    for (const auto& val : ints) {
        result = std::gcd(result, val);
        if (result == 1) {
            return 1;
        }
    }
    return result;
}
static_assert(getGcd(std::array<int, 3>{2, 4, 6}) == 2);

LineFitResult evaluateCandidate(float candidate, float absoluteErrorThreshold, PeakData peaks,
                                std::vector<float> weights) {
    if (peaks.indices.empty() || candidate <= 0.f) {
        return {};
    }

    // Derive harmonic numbers for each peak: k[i] = max(round(peakIndices[i] / candidate), 1)
    std::vector<int> k(peaks.indices.size());
    std::transform(peaks.indices.begin(), peaks.indices.end(), k.begin(), [candidate](int index) {
        return std::max(1, static_cast<int>(std::round(index / candidate)));
    });

    LineFitResult bestFit = {};

    while (k.size() > 1) {
        // TODO explain
        const std::unordered_set<int> kSet{k.begin(), k.end()};
        for (auto divisor : {2, 3}) {
            const auto numDividables = std::accumulate(
                kSet.begin(), kSet.end(), 0,
                [divisor](int acc, int val) { return acc + (val % divisor == 0 ? 1 : 0); });
            if (numDividables >= kSet.size() - 2) {
                return bestFit;
            }
        }

        auto fit = leastSquareFit(k, peaks, weights);

        // Check if it's converged
        const auto allOk =
            std::all_of(fit.absErrors.begin(), fit.absErrors.end(),
                        [absoluteErrorThreshold](float e) { return e < absoluteErrorThreshold; });

        if (allOk /*  || fit.meanSquaredError / bestFit.meanSquaredError > 0.9f */) {
            bestFit = fit;
            break;
        }

        // Find the index with the largest weighted error and remove it
        const auto maxErrorPos = std::distance(
            fit.absErrors.begin(), std::max_element(fit.absErrors.begin(), fit.absErrors.end()));
        k.erase(k.begin() + maxErrorPos);

        const auto kGcd = getGcd(k);
        if (kGcd > 1) {
            // We could multiply the result of the next evaluation by kGcd, or break now and let
            // another, dedicated evaluation find out for itself.
            break;
        }

        peaks.indices.erase(peaks.indices.begin() + maxErrorPos);
        peaks.values.erase(peaks.values.begin() + maxErrorPos);
        weights.erase(weights.begin() + maxErrorPos);

        if (fit.meanSquaredError < bestFit.meanSquaredError)
            bestFit = fit;
    }

    return bestFit;
}

float disambiguateFundamentalIndex(float octaviatedIndex, const std::vector<float>& idealSpectrum,
                                   float minF0) {
    const auto& spec = idealSpectrum;
    // `octaviatedIndex` is the fundamental frequency estimate based on autocorrelation.
    // At the time of writing, the parent commit yields an accuracy histogram where
    // * 96.8% of the estimates are "exact" (within [-50, 50] cents of the ground truth),
    // * 1.5% are an octave too high
    // * 0.6% are an octave too low
    // * 0.13% are an octave and a fifth too low.
    // * other "octaviation" mistakes are less than 1 per mil - we neglect them.
    // The candidates are hence
    const std::array<float, 4> candidates{octaviatedIndex, octaviatedIndex * 2, octaviatedIndex / 2,
                                          octaviatedIndex / 3};
    constexpr std::array<float, 4> probabilities{0.968f, 0.015f, 0.006f, 0.0013f};

    // Here is the idea:
    // 1. get a vector of the peaks in the whitened spectrum: `peakIndices` and `peakValues`.
    // 2. derive a vector of corresponding weights, w[i] = idealSpectrum[peakIndices[i]] /
    // sum(idealSpectrum[peakIndices]) For each of the candidates:
    // 1. derive a vector of harmonic numbers, k[i] = max(round(peakIndices[i] / candidate), 1)
    // 2. Initialize `peakIndexIndices = [0, 1, ..., <num peaks>)`
    //    * If the length or peakIndexIndices is 1 or less, break.
    //    * Fit a line in the least-square sense to get `a` and `b` and get the error vector e[i] =
    //    a*k[peakIndexIndices[i]] + b - peakIndices[peakIndexIndices[i]]
    //    * If the squared error is less than the threshold (TBD), break.
    //    * Remove then entry of `peakIndexIndices` that points to the largest error. Recommence.
    // 3. Get candidate that corresponds to the least error.

    // Step 1: Get peaks from the ideal spectrum
    const auto minCandidate = *std::min_element(candidates.begin(), candidates.end());
    const auto minSearchIndex = std::max(minCandidate * 0.9f, minF0);
    const auto maxSearchIndex =
        static_cast<int>(std::min(20.f * octaviatedIndex, static_cast<float>(spec.size()) / 2));
    const PeakData peaks = getPeaks(spec, minSearchIndex, maxSearchIndex);

    if (peaks.indices.empty()) {
        return octaviatedIndex;  // No peaks found, return original estimate
    }

    // Step 2: Compute weights w[i] = peakValues[i] / sum(peakValues)
    const float sumValues = std::accumulate(peaks.values.begin(), peaks.values.end(), 0.f);
    std::vector<float> weights(peaks.values.size());
    if (sumValues > 0.f) {
        std::transform(peaks.values.begin(), peaks.values.end(), weights.begin(),
                       [sumValues](float v) { return v / sumValues; });
    } else {
        // Fall back to uniform weights if all values are non-positive
        std::fill(weights.begin(), weights.end(), 1.f / weights.size());
    }

    // Step 3: Evaluate each candidate and find the best one
    std::optional<LineFitResult> bestFit;
    for (auto c = 0; c < candidates.size(); ++c) {
        const auto candidate = candidates[c];
        // Skip candidates below the minimum detectable frequency
        if (candidate < minF0) {
            continue;
        }

        const auto absoluteErrorThreshold = candidate / 20.f;
        const LineFitResult candidateFit =
            evaluateCandidate(candidate, absoluteErrorThreshold, peaks, weights);

        const auto squaredErrorThreshold = absoluteErrorThreshold * absoluteErrorThreshold;
        if (!bestFit.has_value() && candidateFit.meanSquaredError < squaredErrorThreshold) {
            // The original estimate looks good already, no need to take risks.
            return octaviatedIndex;
        }

        if (!bestFit.has_value() || candidateFit.meanSquaredError < bestFit->meanSquaredError) {
            bestFit = candidateFit;
        }
    }

    return bestFit.has_value() ? bestFit->slope : octaviatedIndex;
}
}  // namespace

float PitchDetectorImpl::disambiguateEstimate(float priorEstimate,
                                              const std::vector<float>& idealSpectrum) const {
    const auto priorIndex = priorEstimate / _binFreq;
    const auto minF0 = _minFreq / _binFreq;
    return disambiguateFundamentalIndex(priorIndex, idealSpectrum, minF0) * _binFreq;
}

float PitchDetectorImpl::getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const {
    const auto cepstrum = cepstrumData.vec();

    // We're using this for a tuner, so look between 30Hz (a detuned E0 on a bass
    // guitar) and 500Hz (Ukulele high A is 440Hz)
    const auto maxPeriod = 1. / _minFreq;
    const auto minPeriod = 1. / _maxFreq;
    const auto leftmost = static_cast<int>(minPeriod * _sampleRate);
    const auto rightmost = std::min<int>(maxPeriod * _sampleRate, cepstrum.size());
    const auto it = std::max_element(cepstrum.begin() + leftmost, cepstrum.begin() + rightmost);
    auto maxCepstrumIndex = std::distance(cepstrum.begin(), it);
    if (maxCepstrumIndex == 0)
        return 0.f;
    else if (maxCepstrumIndex == cepstrum.size())
        return static_cast<float>(_sampleRate) / maxCepstrumIndex;

    const auto bestIndex =
        takeThisIndexInstead(cepstrum, leftmost, maxCepstrumIndex).value_or(maxCepstrumIndex);
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
      _cepstrumFft(_fftSize),
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
    constexpr auto numHarmonics = 10;
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

    const auto refined = (static_cast<float>(peakBin) + delta) * _binFreq / k;
    return refined;
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

    std::vector<float> dbSpectrum(_fftSize);
    utils::getPowerSpectrum(freq, dbSpectrum);
    std::transform(dbSpectrum.begin(), dbSpectrum.end(), dbSpectrum.begin(),
                   [](float power) { return utils::FastDb(power); });
    assert(utils::isSymmetric(dbSpectrum));
    _logger->Log(dbSpectrum.data(), dbSpectrum.size(), "dbSpectrum");

    // Compute cross-correlation
    getXCorr(_fwdFft, time, freq, _lpWindow);
    _logger->Log(time.data(), time.size(), "xcorr");
    _logger->Log(time.data(), time.size(), "xcorrFlattened",
                 [this, i = 0](float x) mutable { return x / std::max(_windowXcor[i++], 1e-6f); });

    utils::Finally finally([this]() { _logger->EndNewEstimate(nullptr, 0); });

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

    const auto xcorrEstimate = static_cast<float>(_sampleRate) / maxIndex;

    const auto refinedEstimate = refineEstimateBasedOnStrongestHarmonic(dbSpectrum, xcorrEstimate);

    auto idealSpectrum = dbSpectrum;
    toIdealSpectrum(idealSpectrum);

    const auto disambiguatedEstimate = disambiguateEstimate(refinedEstimate, idealSpectrum);

    _logger->EndNewEstimate(nullptr, 0);

    return disambiguatedEstimate;
}

void PitchDetectorImpl::toIdealSpectrum(std::vector<float>& logSpectrum) {
    auto& spec = logSpectrum;

    Aligned<std::vector<float>> cepstrumAligned;
    toCepstrum(spec, _cepstrumFft, cepstrumAligned);

    const std::vector<float>& cepstrum = cepstrumAligned.value;
    std::vector<float> lifteredCepstrum = cepstrum;
    const auto cutoffIndex = std::min<int>(_sampleRate / 2500.f, cepstrum.size());
    std::fill(lifteredCepstrum.begin() + cutoffIndex, lifteredCepstrum.end() - cutoffIndex + 1,
              0.f);

    const std::vector<float> spectrumEnvelope = fromCepstrum(_cepstrumFft, lifteredCepstrum.data());
    _logger->Log(spectrumEnvelope.data(), spectrumEnvelope.size(), "spectrumEnvelope");

    std::transform(spec.begin(), spec.end(), spectrumEnvelope.begin(), spec.begin(),
                   std::minus<float>());

    // Calculate the variance from 5kHz to the Nyquist
    const auto minFreq = 5000.f;
    const auto minBin = static_cast<int>(minFreq / _binFreq);
    const auto N = static_cast<float>(static_cast<int>(spec.size()) - minBin);

    // Expected value E
    const auto E = std::accumulate(spec.begin() + minBin, spec.end(), 0.f,
                                   [](float acc, float val) { return acc + val; }) /
                   N;

    const auto variance =
        std::accumulate(spec.begin(), spec.end(), 0.f,
                        [E](float acc, float val) { return acc + (val - E) * (val - E); }) /
        N;

    const auto stdDev = std::sqrt(variance);
    const auto noiseThreshold = stdDev * 1.5f;

    std::transform(spec.begin(), spec.end(), spec.begin(),
                   [noiseThreshold](float x) { return x - noiseThreshold; });

    _logger->Log(spec.data(), spec.size(), "idealSpectrum");

    assert(utils::isSymmetric(spec));
    assert(utils::isPowerOfTwo(spec.size()));
}

}  // namespace saint
