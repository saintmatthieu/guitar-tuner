#include "AutocorrEstimateDisambiguator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "PitchDetectorLoggerInterface.h"
#include "PitchDetectorTypes.h"
#include "PitchDetectorUtils.h"

namespace saint {
namespace {
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

float AutocorrEstimateDisambiguator::disambiguateEstimate(
    float priorEstimate, const std::vector<float>& idealSpectrum) const {
    const auto priorIndex = priorEstimate / _binFreq;
    const auto minF0 = _minFreq / _binFreq;
    return disambiguateFundamentalIndex(priorIndex, idealSpectrum, minF0) * _binFreq;
}

AutocorrEstimateDisambiguator::AutocorrEstimateDisambiguator(
    int sampleRate, int fftSize, const std::optional<PitchDetectorConfig>& config,
    PitchDetectorLoggerInterface& logger)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _binFreq(static_cast<float>(sampleRate) / _fftSize),
      _cepstrumFft(_fftSize),
      _minFreq(getMinFreq(config)),
      _maxFreq(getMaxFreq(config)) {}

float AutocorrEstimateDisambiguator::process(float xcorrEstimate,
                                             const std::vector<std::complex<float>>& spectrum) {
    std::vector<float> dbSpectrum(_fftSize);
    utils::getPowerSpectrum(spectrum, dbSpectrum);
    std::transform(dbSpectrum.begin(), dbSpectrum.end(), dbSpectrum.begin(),
                   [](float power) { return utils::FastDb(power); });
    assert(utils::isSymmetric(dbSpectrum));
    _logger.Log(dbSpectrum.data(), dbSpectrum.size(), "dbSpectrum");

    auto idealSpectrum = dbSpectrum;
    toIdealSpectrum(idealSpectrum);

    const auto disambiguatedEstimate = disambiguateEstimate(xcorrEstimate, idealSpectrum);

    return disambiguatedEstimate;
}

void AutocorrEstimateDisambiguator::toIdealSpectrum(std::vector<float>& logSpectrum) {
    auto& spec = logSpectrum;

    Aligned<std::vector<float>> cepstrumAligned;
    toCepstrum(spec, _cepstrumFft, cepstrumAligned);

    const std::vector<float>& cepstrum = cepstrumAligned.value;
    std::vector<float> lifteredCepstrum = cepstrum;
    const auto cutoffIndex = std::min<int>(_sampleRate / 2500.f, cepstrum.size());
    std::fill(lifteredCepstrum.begin() + cutoffIndex, lifteredCepstrum.end() - cutoffIndex + 1,
              0.f);

    const std::vector<float> spectrumEnvelope = fromCepstrum(_cepstrumFft, lifteredCepstrum.data());
    _logger.Log(spectrumEnvelope.data(), spectrumEnvelope.size(), "spectrumEnvelope");

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

    _logger.Log(spec.data(), spec.size(), "idealSpectrum");

    assert(utils::isSymmetric(spec));
    assert(utils::isPowerOfTwo(spec.size()));
}

}  // namespace saint
