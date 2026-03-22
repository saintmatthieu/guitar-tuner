#include "AutocorrEstimateDisambiguator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "PitchDetector.h"
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

LineFitResult leastSquareFit(const std::vector<int>& k, const std::vector<PeakModel>& peaks) {
    // Fit a line y = a*x + b to the data points (k[i], peakIndices[i]) for i in activeIndices
    // using weighted least squares.
    const auto n = k.size();
    if (n < 2) {
        return {};
    }

    std::vector<float> x(n), y(n), w(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(k[i]);
        y[i] = static_cast<float>(peaks[i].index);
        w[i] = peaks[i].weight;
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

std::optional<LineFitResult> evaluateCandidate(float candidate, float absoluteErrorThreshold,
                                               std::vector<PeakModel> spectrumModel) {
    if (spectrumModel.empty() || candidate <= 0.f) {
        return {};
    }

    // Derive harmonic numbers for each peak: k[i] = max(round(peakIndices[i] / candidate), 1)
    std::vector<int> k(spectrumModel.size());
    std::transform(spectrumModel.begin(), spectrumModel.end(), k.begin(),
                   [candidate](const PeakModel& peak) {
                       return static_cast<int>(std::round(peak.index / candidate));
                   });

    // If there is a harmonic gap of 10 or more, what comes after is very unlikely harmonics - get
    // rid of it:
    auto lastValidHarmonic = 0;
    for (size_t i = 1; i < k.size(); ++i) {
        if (k[i] - k[i - 1] > 10) {
            break;
        }
        lastValidHarmonic = i;
    }
    k.resize(lastValidHarmonic + 1);

    if (k.size() < 3) {
        // Too risky
        return {};
    }

    std::optional<LineFitResult> bestFit = {};

    while (k.size() > 1) {
        const std::unordered_set<int> kSet{k.begin(), k.end()};

        // For a candidate that's an underestimate by a factor of 2, the peaks that are present
        // will still explain very well that candidate. However, the k values for these cases will
        // look like [2, 4, 6, ...], i.e., most of them will be dividable by 2. Same goes for 3. If
        // we detect such a situation, we abort.
        const auto K = static_cast<int>(kSet.size());
        for (auto divisor : {2, 3}) {
            const auto numDividables = std::accumulate(
                kSet.begin(), kSet.end(), 0,
                [divisor](int acc, int val) { return acc + (val % divisor == 0 ? 1 : 0); });
            const auto expectedIfCorrect = K / divisor;
            const auto expectedIfIncorrect = K;
            const auto closerToIncorrect =
                expectedIfIncorrect - numDividables < std::abs(expectedIfCorrect - numDividables);
            if (closerToIncorrect) {
                return bestFit;
            }
        }

        // Next caveat: the candidate is an overestimate by a factor of 2, then the k values will
        // tend to look like [1, 1, 2, 2, ...]. The least square fit in such cases isn't that bad,
        // actually, so just relying on this isn't so robust. Instead, let's just look at how many
        // duplicates there are ...
        if (1. * kSet.size() / k.size() < 0.9) {
            return bestFit;
        }

        const auto fit = leastSquareFit(k, spectrumModel);

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

        spectrumModel.erase(spectrumModel.begin() + maxErrorPos);

        if (!bestFit || fit.meanSquaredError < bestFit->meanSquaredError)
            bestFit = fit;
    }

    return bestFit;
}

float disambiguateFundamentalIndex(float octaviatedIndex,
                                   const std::vector<PeakModel>& spectrumModel, float minF0,
                                   std::optional<float> constraintIndex) {
    // `octaviatedIndex` is the fundamental frequency estimate based on autocorrelation.
    // At the time of writing, the parent commit yields an accuracy histogram where
    // * 96.8% of the estimates are "exact" (within [-50, 50] cents of the ground truth),
    // * 1.5% are an octave too high
    // * 0.6% are an octave too low
    // * 0.13% are an octave and a fifth too low.
    // * other "octaviation" mistakes are less than 1 per mil - we neglect them.
    // The candidates are hence
    const std::array<float, 4> allCandidates{octaviatedIndex, octaviatedIndex * 2,
                                             octaviatedIndex / 2, octaviatedIndex / 3};

    // If constrained, filter candidates to those within a major third of the constraint
    std::vector<float> candidates;
    for (const auto& c : allCandidates) {
        if (!constraintIndex.has_value()) {
            candidates.push_back(c);
        } else {
            const auto minIndex = constraintIndex.value() / majorThirdRatio;
            const auto maxIndex = constraintIndex.value() * majorThirdRatio;
            if (c >= minIndex && c <= maxIndex) {
                candidates.push_back(c);
            }
        }
    }

    // If no candidates remain after filtering, just return the original estimate
    if (candidates.empty()) {
        return octaviatedIndex;
    }

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

    // Step 3: Evaluate each candidate and find the best one
    std::optional<LineFitResult> bestFit;
    auto bestCandidate = 0.f;
    for (auto c = 0; c < candidates.size(); ++c) {
        const auto candidate = candidates[c];
        // Skip candidates below the minimum detectable frequency
        if (candidate < minF0) {
            continue;
        }

        const auto absoluteErrorThreshold = candidate / 20.f;
        const std::optional<LineFitResult> candidateFit =
            evaluateCandidate(candidate, absoluteErrorThreshold, spectrumModel);

        if (candidateFit.has_value() &&
            (!bestFit.has_value() || candidateFit->meanSquaredError < bestFit->meanSquaredError)) {
            bestFit = candidateFit;
            bestCandidate = candidate;
        }
    }

    return bestCandidate;  // Could be 0
}
}  // namespace

float AutocorrEstimateDisambiguator::process(float priorEstimate,
                                             const std::vector<PeakModel>& spectrumModel,
                                             std::optional<float> constraint) const {
    const auto priorIndex = priorEstimate / _binFreq;
    const auto minF0 = _minFreq / _binFreq;
    const auto constraintIndex =
        constraint.has_value() ? std::optional<float>(constraint.value() / _binFreq) : std::nullopt;
    return disambiguateFundamentalIndex(priorIndex, spectrumModel, minF0, constraintIndex) *
           _binFreq;
}

AutocorrEstimateDisambiguator::AutocorrEstimateDisambiguator(int sampleRate, int fftSize,
                                                             Tuning tuning,
                                                             PitchDetectorLoggerInterface& logger)
    : _logger(logger),
      _binFreq(static_cast<float>(sampleRate) / fftSize),
      _minFreq(getMinFreq(tuning)),
      _maxFreq(getMaxFreq(tuning)) {}

}  // namespace saint
