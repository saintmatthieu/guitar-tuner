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

LineFitResult evaluateCandidate(float candidate, float absoluteErrorThreshold,
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

    LineFitResult bestFit = {};

    while (k.size() > 1) {
        // std::unordered_map<int /* k */, float /* avg of weights */> kMap;
        // for (auto i = 0; i < spectrumModel.size(); ++i) {
        //     const auto k = static_cast<int>(std::round(spectrumModel[i].index / candidate));
        //     kMap[k] += spectrumModel[i]
        // }

        const std::unordered_set<int> kSet{k.begin(), k.end()};

        const auto sumOfWeights =
            std::accumulate(spectrumModel.begin(), spectrumModel.end(), 0.f,
                            [](float acc, const PeakModel& pm) { return acc + pm.weight; });

        // For a candidate that's an underestimate by a factor of 2, the peaks that are present
        // will still explain very well that candidate. However, the k values for these cases will
        // look like [2, 4, 6, ...], i.e., most of them will be dividable by 2. Same goes for 3. If
        // we detect such a situation, we abort.
        for (auto divisor : {2, 3}) {
            const auto numDividables = std::accumulate(
                kSet.begin(), kSet.end(), 0,
                [divisor](int acc, int val) { return acc + (val % divisor == 0 ? 1 : 0); });
            if (numDividables >= kSet.size() - 2) {
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

        auto fit = leastSquareFit(k, spectrumModel);

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

        if (fit.meanSquaredError < bestFit.meanSquaredError)
            bestFit = fit;
    }

    return bestFit;
}

float toBark(int bin, float binFrequency) {
    const auto f = bin * binFrequency;
    constexpr auto den2 = 7500.f * 7500.f;
    return 13.f * std::atan(0.00076f * f) + 3.5f * std::atan(f * f / den2);
}

template <WindowType W>
std::vector<PeakModel> toSpectrumModel(std::vector<float> dbSpectrum, double binResolution,
                                       double binFrequency,
                                       std::vector<float>* fullIdeal = nullptr) {
    constexpr auto maxNumPeaks = 20;
    std::vector<PeakModel> spectrumModel;
    spectrumModel.reserve(maxNumPeaks);
    if (fullIdeal) {
        fullIdeal->resize(dbSpectrum.size());
        std::fill(fullIdeal->begin(), fullIdeal->end(), -1000.f);
    }

    const auto numBinsForFit =
        static_cast<int>(utils::mainLobeWidth<W>() / binResolution / 2) * 2 + 1;
    std::vector<float> idealLobeScratch(numBinsForFit);
    std::vector<float> lobeFittingWeights(numBinsForFit);

    std::vector<int> peakIndices;
    for (auto i = 1; i < dbSpectrum.size() - 1; ++i) {
        if (dbSpectrum[i] > dbSpectrum[i - 1] && dbSpectrum[i] > dbSpectrum[i + 1]) {
            peakIndices.push_back(i);
        }
    }

    // Perceptual masking: remove imperceptible peaks with a mask of -3.5dB per octave:
    {
        constexpr auto dbPerBark = 20.f;
        std::vector<int> indicesToRemove;
        for (size_t i = 0; i < peakIndices.size(); ++i) {
            const auto pi = peakIndices[i];
            const auto level = dbSpectrum[pi];
            bool masked = false;
            for (size_t j = 0; j < peakIndices.size() && !masked; ++j) {
                if (i == j)
                    continue;
                const auto pj = peakIndices[j];
                if (dbSpectrum[pj] <= level)
                    continue;
                const auto barks = std::abs(toBark(pi, binFrequency) - toBark(pj, binFrequency));
                if (level < dbSpectrum[pj] - dbPerBark * barks) {
                    masked = true;
                    break;
                }
            }
            if (masked)
                indicesToRemove.push_back(static_cast<int>(i));
        }
        for (auto it = indicesToRemove.rbegin(); it != indicesToRemove.rend(); ++it)
            peakIndices.erase(peakIndices.begin() + *it);
    }

    // 0. Create a vector of indices [0, 1, ..., dbSpectrum.size()-1].
    // 1. Find `dbSpectrum`'s global max.
    // 2. Find the troughs left and right of it.
    // 3. Use a quadratic fit to refine the index estimate of the peak.
    // 4. Evaluate the ideal lobe at the fractional bins of the peak.
    // 5. Calculate the mean of the square errors between ideal and actual peak.
    // 6. If the error is larger than a threshold (to be found and tuned), finish.
    // 7. Add an entry to `spectrumModel`.
    // 8. Erase dbSpectrum and index vector entries of the peak.
    // 9. Go back to 1.

    // 0.
    while (!peakIndices.empty() && spectrumModel.size() <= maxNumPeaks) {
        // 1.
        const auto maxIt =
            std::max_element(peakIndices.begin(), peakIndices.end(),
                             [&dbSpectrum](int a, int b) { return dbSpectrum[a] < dbSpectrum[b]; });
        const auto maxIndex = *maxIt;
        const auto maxValue = dbSpectrum[maxIndex];
        peakIndices.erase(maxIt);

        // 2.
        const auto beginIndex = maxIndex - (numBinsForFit - 1) / 2;
        const auto endIndex = beginIndex + numBinsForFit;
        if (beginIndex < 0 || endIndex > dbSpectrum.size()) {
            break;
        }

        // 3.
        float refinedIndex = maxIndex;
        float refinedValue = maxValue;
        if (maxIndex > 0 && maxIndex < dbSpectrum.size() - 1) {
            const float y[3] = {dbSpectrum[maxIndex - 1], dbSpectrum[maxIndex],
                                dbSpectrum[maxIndex + 1]};
            refinedIndex += utils::quadFit(y, &refinedValue);
        }

        // 4.
        for (auto i = beginIndex; i < endIndex; ++i) {
            const auto linearValue = utils::mainLobeAt<W>((i - refinedIndex) * binResolution);
            const auto db = utils::FastDb(linearValue * linearValue) + refinedValue;
            idealLobeScratch[i - beginIndex] = db;
            lobeFittingWeights[i - beginIndex] =
                1 - std::pow((i - refinedIndex) * 2 / numBinsForFit, 16);
        }
        if (fullIdeal) {
            for (auto i = beginIndex; i < endIndex; ++i) {
                (*fullIdeal)[i] = idealLobeScratch[i - beginIndex];
            }
        }

        // 5.
        const std::vector<float> actualLobe{dbSpectrum.begin() + beginIndex,
                                            dbSpectrum.begin() + endIndex};
        auto squareSum = 0.f;
        for (auto i = 0; i < numBinsForFit; ++i) {
            const auto error = idealLobeScratch[i] - actualLobe[i];
            squareSum += error * error * lobeFittingWeights[i];
        }
        const auto rmsError = std::sqrt(squareSum / numBinsForFit);

        // 6.
        constexpr auto coef = -0.5157894736842106f;
        const auto weight = std::min(-coef / (coef + rmsError), 1.f);
        if (weight > 0.1)
            spectrumModel.push_back({refinedIndex, refinedValue, weight});
    }

    // sort peaks by index
    std::sort(spectrumModel.begin(), spectrumModel.end(),
              [](const PeakModel& a, const PeakModel& b) { return a.index < b.index; });

    return spectrumModel;
}

float disambiguateFundamentalIndex(float octaviatedIndex,
                                   const std::vector<PeakModel>& spectrumModel, float minF0,
                                   std::optional<float> constraintIndex) {
    const auto& spec = spectrumModel;
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
        const LineFitResult candidateFit =
            evaluateCandidate(candidate, absoluteErrorThreshold, spec);

        const auto squaredErrorThreshold = absoluteErrorThreshold * absoluteErrorThreshold;
        if (!bestFit.has_value() && candidateFit.meanSquaredError < squaredErrorThreshold) {
            // The original estimate looks good already, no need to take risks.
            return octaviatedIndex;
        }

        if (!bestFit.has_value() || candidateFit.meanSquaredError < bestFit->meanSquaredError) {
            bestFit = candidateFit;
            bestCandidate = candidate;
        }
    }

    return bestCandidate > 0.f ? bestCandidate : octaviatedIndex;
}
}  // namespace

float AutocorrEstimateDisambiguator::disambiguateEstimate(
    float priorEstimate, const std::vector<PeakModel>& spectrumModel,
    std::optional<float> constraint) const {
    const auto priorIndex = priorEstimate / _binFreq;
    const auto minF0 = _minFreq / _binFreq;
    const auto constraintIndex =
        constraint.has_value() ? std::optional<float>(constraint.value() / _binFreq) : std::nullopt;
    return disambiguateFundamentalIndex(priorIndex, spectrumModel, minF0, constraintIndex) *
           _binFreq;
}

AutocorrEstimateDisambiguator::AutocorrEstimateDisambiguator(
    int sampleRate, int windowSize, int fftSize, const std::optional<PitchDetectorConfig>& config,
    PitchDetectorLoggerInterface& logger)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _binResolution(1. * windowSize / fftSize),
      _binFreq(static_cast<float>(sampleRate) / _fftSize),
      _cepstrumFft(_fftSize),
      _minFreq(getMinFreq(config)),
      _maxFreq(getMaxFreq(config)) {}

float AutocorrEstimateDisambiguator::process(float xcorrEstimate,
                                             const std::vector<float>& dbSpectrum,
                                             std::optional<float> constraint) {
    auto idealSpectrum = dbSpectrum;
    toIdealSpectrum(idealSpectrum);

    std::vector<float> fullIdeal;
    const std::vector<PeakModel> spectrumModel = toSpectrumModel<kWindowType>(
        {dbSpectrum.begin(), dbSpectrum.begin() + dbSpectrum.size() / 2}, _binResolution, _binFreq,
        &fullIdeal);
    std::vector<float> spectrumModelIndices(spectrumModel.size());
    std::transform(spectrumModel.begin(), spectrumModel.end(), spectrumModelIndices.begin(),
                   [](const PeakModel& pm) { return pm.index; });
    std::vector<float> spectrumModelValues(spectrumModel.size());
    std::transform(spectrumModel.begin(), spectrumModel.end(), spectrumModelValues.begin(),
                   [](const PeakModel& pm) { return pm.value; });
    std::vector<float> spectrumModelWeights(spectrumModel.size());
    std::transform(spectrumModel.begin(), spectrumModel.end(), spectrumModelWeights.begin(),
                   [](const PeakModel& pm) { return pm.weight; });
    _logger.Log(spectrumModelIndices.data(), spectrumModelIndices.size(), "spectrumModelIndices");
    _logger.Log(spectrumModelValues.data(), spectrumModelValues.size(), "spectrumModelValues");
    _logger.Log(spectrumModelWeights.data(), spectrumModelWeights.size(), "spectrumModelWeights");
    _logger.Log(fullIdeal.data(), fullIdeal.size(), "fullIdealSpectrum");

    const auto disambiguatedEstimate =
        disambiguateEstimate(xcorrEstimate, spectrumModel, constraint);

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
