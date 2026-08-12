#include "AutocorrEstimateDisambiguator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

#include "PitchDetectorLoggerInterface.h"
#include "PitchDetectorTypes.h"
#include "PitchDetectorUtils.h"

namespace saint {
namespace {
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
}  // namespace

void AutocorrEstimateDisambiguator::getPeaks(const std::vector<float>& spectrum, int minIndex,
                                             int maxIndex, PeakData& out) {
    out.indices.clear();
    out.values.clear();

    for (int i = minIndex; i < maxIndex - 1; ++i) {
        if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.f) {
            out.indices.push_back(i);
            out.values.push_back(spectrum[i]);
        }
    }

    if (out.indices.size() == 1) {
        return;
    }

    // Remove peaks that aren't looking good because of interference with noise or another peak
    // that's too close.
    constexpr auto minDiffDb = 10.f;
    _peakRemove.clear();
    for (size_t i = 0; i < out.indices.size(); ++i) {
        auto leftTroughIndex = out.indices[i];
        while (leftTroughIndex > 0 && spectrum[leftTroughIndex - 1] < spectrum[leftTroughIndex]) {
            --leftTroughIndex;
        }
        if (leftTroughIndex == 0 ||
            spectrum[out.indices[i]] - spectrum[leftTroughIndex] < minDiffDb) {
            _peakRemove.push_back(i);
            continue;
        }

        auto rightTroughIndex = out.indices[i];
        while (rightTroughIndex + 1 < maxIndex &&
               spectrum[rightTroughIndex + 1] < spectrum[rightTroughIndex]) {
            ++rightTroughIndex;
        }
        if (rightTroughIndex + 1 == maxIndex ||
            spectrum[out.indices[i]] - spectrum[rightTroughIndex] < minDiffDb) {
            _peakRemove.push_back(i);
        }
    }
    for (auto it = _peakRemove.rbegin(); it != _peakRemove.rend(); ++it) {
        out.indices.erase(out.indices.begin() + *it);
        out.values.erase(out.values.begin() + *it);
    }
}

float AutocorrEstimateDisambiguator::leastSquareFit(const std::vector<int>& k,
                                                    const PeakData& peaks,
                                                    const std::vector<float>& weights,
                                                    std::vector<float>& absErrorsOut) {
    // Fit a line y = a*x through the data points (k[i], peakIndices[i]) using weighted least
    // squares (a = sum(x*y*w) / sum(x*x*w), matching utils::leastSquareFit), and return the
    // weighted mean squared error while writing the per-point absolute residuals to absErrorsOut.
    const auto n = k.size();
    if (n < 2) {
        absErrorsOut.clear();
        return std::numeric_limits<float>::max();
    }

    float num = 0.f;
    float den = 0.f;
    for (size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(k[i]);
        const float y = static_cast<float>(peaks.indices[i]);
        const float w = weights[i];
        num += x * y * w;
        den += x * x * w;
    }
    const float a = num / den;

    float meanSquaredError = 0.f;
    absErrorsOut.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(k[i]);
        const float y = static_cast<float>(peaks.indices[i]);
        const float residual = a * x - y;
        absErrorsOut[i] = std::abs(residual);
        meanSquaredError += weights[i] * residual * residual;
    }
    meanSquaredError /= n;

    return meanSquaredError;
}

float AutocorrEstimateDisambiguator::evaluateCandidate(float candidate,
                                                       float absoluteErrorThreshold,
                                                       const PeakData& peaks,
                                                       const std::vector<float>& weights) {
    if (peaks.indices.empty() || candidate <= 0.f) {
        return std::numeric_limits<float>::max();
    }

    // Mutable working copies (the pruning loop erases entries); reusing members keeps the
    // copies allocation-free after warm-up.
    _peaksWork.indices.assign(peaks.indices.begin(), peaks.indices.end());
    _peaksWork.values.assign(peaks.values.begin(), peaks.values.end());
    _weightsWork.assign(weights.begin(), weights.end());

    // Derive harmonic numbers for each peak: k[i] = max(round(peakIndices[i] / candidate), 1)
    _k.resize(_peaksWork.indices.size());
    std::transform(_peaksWork.indices.begin(), _peaksWork.indices.end(), _k.begin(),
                   [candidate](int index) {
                       return std::max(1, static_cast<int>(std::round(index / candidate)));
                   });

    float bestMse = std::numeric_limits<float>::max();

    while (_k.size() > 1) {
        // Count the DISTINCT k values and how many are divisible by 2 / 3, via a sorted copy
        // (a node-based set would allocate per element on every call). Equivalent to the old
        // unordered_set: the integer counts are identical.
        _kSorted.assign(_k.begin(), _k.end());
        std::sort(_kSorted.begin(), _kSorted.end());
        size_t numDistinct = 0;
        int numDivisibleBy2 = 0;
        int numDivisibleBy3 = 0;
        for (size_t i = 0; i < _kSorted.size(); ++i) {
            if (i == 0 || _kSorted[i] != _kSorted[i - 1]) {
                ++numDistinct;
                if (_kSorted[i] % 2 == 0) ++numDivisibleBy2;
                if (_kSorted[i] % 3 == 0) ++numDivisibleBy3;
            }
        }

        // For a candidate that's an underestimate by a factor of 2, the peaks that are present
        // will still explain very well that candidate. However, the k values for these cases will
        // look like [2, 4, 6, ...], i.e., most of them will be dividable by 2. Same goes for 3. If
        // we detect such a situation, we abort.
        if (numDivisibleBy2 >= numDistinct - 2) {
            return bestMse;
        }
        if (numDivisibleBy3 >= numDistinct - 2) {
            return bestMse;
        }

        // Next caveat: the candidate is an overestimate by a factor of 2, then the k values will
        // tend to look like [1, 1, 2, 2, ...]. The least square fit in such cases isn't that bad,
        // actually, so just relying on this isn't so robust. Instead, let's just look at how many
        // duplicates there are ...
        if (1. * numDistinct / _k.size() < 0.9) {
            return bestMse;
        }

        const float mse = leastSquareFit(_k, _peaksWork, _weightsWork, _absErrors);

        // Check if it's converged
        const auto allOk =
            std::all_of(_absErrors.begin(), _absErrors.end(),
                        [absoluteErrorThreshold](float e) { return e < absoluteErrorThreshold; });

        if (allOk) {
            bestMse = mse;
            break;
        }

        // Find the index with the largest weighted error and remove it
        const auto maxErrorPos = std::distance(
            _absErrors.begin(), std::max_element(_absErrors.begin(), _absErrors.end()));
        _k.erase(_k.begin() + maxErrorPos);

        const auto kGcd = getGcd(_k);
        if (kGcd > 1) {
            // We could multiply the result of the next evaluation by kGcd, or break now and let
            // another, dedicated evaluation find out for itself.
            break;
        }

        _peaksWork.indices.erase(_peaksWork.indices.begin() + maxErrorPos);
        _peaksWork.values.erase(_peaksWork.values.begin() + maxErrorPos);
        _weightsWork.erase(_weightsWork.begin() + maxErrorPos);

        if (mse < bestMse)
            bestMse = mse;
    }

    return bestMse;
}

float AutocorrEstimateDisambiguator::disambiguateFundamentalIndex(
    float octaviatedIndex, const std::vector<float>& idealSpectrum, float minF0,
    std::optional<float> constraintIndex) {
    const auto& spec = idealSpectrum;
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
    _candidates.clear();
    for (const auto& c : allCandidates) {
        if (!constraintIndex.has_value()) {
            _candidates.push_back(c);
        } else {
            const auto minIndex = constraintIndex.value() / majorThirdRatio;
            const auto maxIndex = constraintIndex.value() * majorThirdRatio;
            if (c >= minIndex && c <= maxIndex) {
                _candidates.push_back(c);
            }
        }
    }

    // If no candidates remain after filtering, just return the original estimate
    if (_candidates.empty()) {
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

    // Step 1: Get peaks from the ideal spectrum
    const auto minCandidate = *std::min_element(_candidates.begin(), _candidates.end());
    const auto minSearchIndex = std::max(minCandidate * 0.9f, minF0);
    const auto maxSearchIndex =
        static_cast<int>(std::min(20.f * octaviatedIndex, static_cast<float>(spec.size()) / 2));
    getPeaks(spec, minSearchIndex, maxSearchIndex, _peaks);

    if (_peaks.indices.empty()) {
        return octaviatedIndex;  // No peaks found, return original estimate
    }

    // Step 2: Compute weights w[i] = peakValues[i] / sum(peakValues)
    const float sumValues = std::accumulate(_peaks.values.begin(), _peaks.values.end(), 0.f);
    _weights.resize(_peaks.values.size());
    if (sumValues > 0.f) {
        std::transform(_peaks.values.begin(), _peaks.values.end(), _weights.begin(),
                       [sumValues](float v) { return v / sumValues; });
    } else {
        // Fall back to uniform weights if all values are non-positive
        std::fill(_weights.begin(), _weights.end(), 1.f / _weights.size());
    }

    // Step 3: Evaluate each candidate and find the best one
    bool haveBest = false;
    float bestMseValue = 0.f;
    auto bestCandidate = 0.f;
    for (auto c = 0; c < _candidates.size(); ++c) {
        const auto candidate = _candidates[c];
        // Skip candidates below the minimum detectable frequency
        if (candidate < minF0) {
            continue;
        }

        const auto absoluteErrorThreshold = candidate / 20.f;
        const float candidateMse =
            evaluateCandidate(candidate, absoluteErrorThreshold, _peaks, _weights);

        const auto squaredErrorThreshold = absoluteErrorThreshold * absoluteErrorThreshold;
        if (!haveBest && candidateMse < squaredErrorThreshold) {
            // The original estimate looks good already, no need to take risks.
            return octaviatedIndex;
        }

        if (!haveBest || candidateMse < bestMseValue) {
            haveBest = true;
            bestMseValue = candidateMse;
            bestCandidate = candidate;
        }
    }

    return bestCandidate > 0.f ? bestCandidate : octaviatedIndex;
}

// Fraction of whitened-spectrum peak energy lying on the higher-harmonic (k>=2)
// comb of f0Index. A pure tone has all its energy at k=1 -> ~0; broadband noise and
// inharmonic/octave-misplaced locks scatter energy off-comb -> low; a genuine note
// puts strong, comb-aligned energy in its overtones -> high.
float AutocorrEstimateDisambiguator::computeHarmonicity(const std::vector<float>& spec,
                                                        float f0Index, float minF0Index) {
    if (f0Index <= 0.f) {
        return 0.f;
    }
    const int minSearch = std::max(1, static_cast<int>(minF0Index));
    const int maxSearch =
        std::min(static_cast<int>(10.f * f0Index), static_cast<int>(spec.size()) / 2);
    if (maxSearch <= minSearch) {
        return 0.f;
    }
    getPeaks(spec, minSearch, maxSearch, _peaks);
    // Tolerance for "peak sits on harmonic k": a fraction of the harmonic spacing,
    // floored so low f0 (few bins per harmonic) isn't impossibly strict.
    const float tol = std::max(1.5f, f0Index * 0.1f);
    float totalEnergy = 0.f;
    float overtoneEnergy = 0.f;
    for (size_t i = 0; i < _peaks.indices.size(); ++i) {
        const float value = _peaks.values[i];
        totalEnergy += value;
        const int k = std::max(1, static_cast<int>(std::round(_peaks.indices[i] / f0Index)));
        if (k >= 2 && std::abs(_peaks.indices[i] - k * f0Index) <= tol) {
            overtoneEnergy += value;
        }
    }
    return totalEnergy > 0.f ? overtoneEnergy / totalEnergy : 0.f;
}

// Fraction of harmonic-comb energy lying on the *odd* lines of the f0/2 comb — the
// half-integer multiples of f0 (1.5*f0, 2.5*f0, ...), i.e. the odd harmonics of f0/2.
// A genuine note at f0 has no energy there; a note whose true fundamental is f0/2 (locked
// an octave high because f0/2 is below the search range) shows them strongly. Returns 0
// when f0/2 is in range (>= minF0Index) — there the octave disambiguation handles the
// choice — or when f0 is invalid.
float AutocorrEstimateDisambiguator::computeSubharmonicScore(const std::vector<float>& spec,
                                                             float f0Index, float minF0Index) {
    const float halfF0 = 0.5f * f0Index;
    if (halfF0 <= 0.f || halfF0 >= minF0Index) {
        return 0.f;  // sub-octave in range (or invalid): not an out-of-range octave error
    }
    // Walk the f0/2 comb directly rather than via getPeaks: the half-integer (odd-m) lines can
    // be weak, and getPeaks' prominence pruning would drop them intermittently. At each comb
    // line m*halfF0 take the strongest whitened-spectrum bin in a narrow window; the whitened
    // spectrum is already "dB above the envelope and noise floor", so a non-positive value
    // means no real energy there. Even m -> a harmonic of f0; odd m -> a half-integer multiple
    // of f0 (an odd harmonic of f0/2), which only carries energy if the true fundamental is
    // f0/2. m starts at 2 (= f0): the m=1 line at f0/2 sits below minFreq where low-frequency
    // contamination is unreliable, so it is excluded — the robust evidence is 1.5*f0, 2.5*f0, …
    const int maxBin = static_cast<int>(spec.size()) / 2;
    const int win = std::max(1, static_cast<int>(halfF0 * 0.1f));
    float intEnergy = 0.f;
    float oddEnergy = 0.f;
    for (int m = 2; m <= 20; ++m) {
        const int center = static_cast<int>(std::lround(m * halfF0));
        if (center >= maxBin) {
            break;
        }
        float lineEnergy = 0.f;
        for (int b = std::max(1, center - win); b <= center + win && b < maxBin; ++b) {
            lineEnergy = std::max(lineEnergy, spec[b]);
        }
        if (lineEnergy <= 0.f) {
            continue;
        }
        (m % 2 == 0 ? intEnergy : oddEnergy) += lineEnergy;
    }
    const float total = intEnergy + oddEnergy;
    return total > 0.f ? oddEnergy / total : 0.f;
}

float AutocorrEstimateDisambiguator::disambiguateEstimate(float priorEstimate,
                                                          const std::vector<float>& idealSpectrum,
                                                          std::optional<float> constraint) {
    const auto priorIndex = priorEstimate / _binFreq;
    const auto minF0 = _minFreq / _binFreq;
    const auto constraintIndex =
        constraint.has_value() ? std::optional<float>(constraint.value() / _binFreq) : std::nullopt;
    return disambiguateFundamentalIndex(priorIndex, idealSpectrum, minF0, constraintIndex) *
           _binFreq;
}

AutocorrEstimateDisambiguator::AutocorrEstimateDisambiguator(
    int sampleRate, int fftSize, Tuning tuning, PitchDetectorLoggerInterface& logger,
    int minFreqSemitoneOffset)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _binFreq(static_cast<float>(sampleRate) / _fftSize),
      _cepstrumFft(_fftSize),
      _minFreq(getMinFreq(tuning, minFreqSemitoneOffset)),
      _maxFreq(getMaxFreq(tuning)) {
    _idealSpectrum.resize(_fftSize);
    _cepstrumAligned.value.resize(_fftSize);
    _lifteredCepstrum.resize(_fftSize);
    _spectrumEnvelope.resize(_fftSize);

    // Reserve peak-fitting scratch up front (peak count is bounded by the half-spectrum)
    // so even the first block allocates nothing.
    const auto maxPeaks = static_cast<size_t>(_fftSize) / 2;
    _peaks.indices.reserve(maxPeaks);
    _peaks.values.reserve(maxPeaks);
    _peakRemove.reserve(maxPeaks);
    _candidates.reserve(4);
    _weights.reserve(maxPeaks);
    _peaksWork.indices.reserve(maxPeaks);
    _peaksWork.values.reserve(maxPeaks);
    _weightsWork.reserve(maxPeaks);
    _k.reserve(maxPeaks);
    _kSorted.reserve(maxPeaks);
    _absErrors.reserve(maxPeaks);
}

float AutocorrEstimateDisambiguator::process(float xcorrEstimate,
                                             const std::vector<float>& dbSpectrum,
                                             std::optional<float> constraint, float* harmonicityOut,
                                             float* subharmonicScoreOut) {
    std::copy(dbSpectrum.begin(), dbSpectrum.end(), _idealSpectrum.begin());
    toIdealSpectrum(_idealSpectrum);

    const auto disambiguatedEstimate =
        disambiguateEstimate(xcorrEstimate, _idealSpectrum, constraint);

    if (harmonicityOut) {
        *harmonicityOut = computeHarmonicity(_idealSpectrum, disambiguatedEstimate / _binFreq,
                                             _minFreq / _binFreq);
    }

    if (subharmonicScoreOut) {
        *subharmonicScoreOut = computeSubharmonicScore(
            _idealSpectrum, disambiguatedEstimate / _binFreq, _minFreq / _binFreq);
    }

    return disambiguatedEstimate;
}

void AutocorrEstimateDisambiguator::toIdealSpectrum(std::vector<float>& logSpectrum) {
    auto& spec = logSpectrum;

    toCepstrum(spec, _cepstrumFft, _cepstrumAligned);

    const std::vector<float>& cepstrum = _cepstrumAligned.value;
    std::copy(cepstrum.begin(), cepstrum.end(), _lifteredCepstrum.begin());
    const auto cutoffIndex = std::min<int>(_sampleRate / 2500.f, cepstrum.size());
    std::fill(_lifteredCepstrum.begin() + cutoffIndex, _lifteredCepstrum.end() - cutoffIndex + 1,
              0.f);

    fromCepstrum(_cepstrumFft, _lifteredCepstrum.data(), _spectrumEnvelope);
    _logger.Log(_spectrumEnvelope.data(), _spectrumEnvelope.size(), "spectrumEnvelope");

    std::transform(spec.begin(), spec.end(), _spectrumEnvelope.begin(), spec.begin(),
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
