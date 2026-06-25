#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "Cepstrum.h"
#include "PitchDetectorTypes.h"
#include "RealFft.h"

class PitchDetectorLoggerInterface;

namespace saint {
class AutocorrEstimateDisambiguator {
   public:
    // `minFreqSemitoneOffset` sets the lower bound of the octave-disambiguation search,
    // mirroring getMinFreq (default -3, i.e. three semitones below the tuning's lowest note).
    // Keep it equal to the offset used to build the rest of the pipeline so the search range
    // is consistent end-to-end.
    AutocorrEstimateDisambiguator(int sampleRate, int fftSize, Tuning tuning,
                                  PitchDetectorLoggerInterface& logger,
                                  int minFreqSemitoneOffset = defaultMinFreqSemitoneOffset);

    // If harmonicityOut is non-null, it receives a [0,1] harmonicity score for the
    // returned estimate: the fraction of whitened-spectrum peak energy that lies on
    // the higher-harmonic (k>=2) comb of the estimate. Low for pure tones, broadband
    // noise and inharmonic/octave-misplaced locks; high for genuine harmonic notes.
    float process(float xcorrEstimate, const std::vector<float>& dbSpectrum,
                  std::optional<float> constraint = std::nullopt, float* harmonicityOut = nullptr);

    // Octave-disambiguates a below-minFreq "double-band" estimate (Hz) with minF0 lowered to the
    // low-band floor (lowBandMinFreqRatio * minFreq), returning the resulting frequency (Hz). A
    // genuine low fundamental is returned ~unchanged; the sub-harmonic of an in-range note is
    // corrected upward (~2x) by the comb analysis. The caller treats "returned ~= input" as a
    // confirmed low note and an upward correction as a reject. Whitens dbSpectrum like process().
    // This reuses the divisibility check that catches sub-multiples, which a plain harmonicity
    // score cannot — a sub-octave's harmonic comb is a superset of the true comb, so it would
    // score at least as well as (often better than) the true fundamental.
    float disambiguateLowBand(float estimate, const std::vector<float>& dbSpectrum);

   private:
    // Peaks found in the whitened spectrum (parallel index/value arrays).
    struct PeakData {
        std::vector<int> indices;
        std::vector<float> values;
    };

    float disambiguateEstimate(float priorEstimate, const std::vector<float>& idealSpectrum,
                               std::optional<float> constraint);
    float getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const;
    void toIdealSpectrum(std::vector<float>& logSpectrum);
    // Peak-fitting helpers. These are members (rather than free functions) so they can
    // reuse the scratch buffers below and allocate nothing on the audio thread.
    void getPeaks(const std::vector<float>& spectrum, int minIndex, int maxIndex, PeakData& out);
    float disambiguateFundamentalIndex(float octaviatedIndex,
                                       const std::vector<float>& idealSpectrum, float minF0,
                                       std::optional<float> constraintIndex);
    // Weighted least-squares slope fit; returns the mean squared error and writes the
    // per-point absolute residuals into absErrorsOut.
    float leastSquareFit(const std::vector<int>& k, const PeakData& peaks,
                         const std::vector<float>& weights, std::vector<float>& absErrorsOut);
    // Returns the best-fit mean squared error for `candidate` (max if no usable fit).
    float evaluateCandidate(float candidate, float absoluteErrorThreshold, const PeakData& peaks,
                            const std::vector<float>& weights);
    float computeHarmonicity(const std::vector<float>& spec, float f0Index, float minF0Index);

    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    const float _binFreq;
    RealFft _cepstrumFft;
    const float _minFreq;
    const float _maxFreq;

    // Reused full-spectrum scratch, so process() allocates nothing on the audio thread.
    std::vector<float> _idealSpectrum;       // whitened working spectrum
    Aligned<std::vector<float>> _cepstrumAligned;
    std::vector<float> _lifteredCepstrum;
    std::vector<float> _spectrumEnvelope;

    // Reused peak-fitting scratch (clear()/assign keep capacity -> no per-call allocation).
    PeakData _peaks;             // getPeaks output (disambiguation, then harmonicity)
    std::vector<int> _peakRemove;
    std::vector<float> _candidates;
    std::vector<float> _weights;
    PeakData _peaksWork;         // evaluateCandidate's mutable working copy of _peaks
    std::vector<float> _weightsWork;
    std::vector<int> _k;
    std::vector<int> _kSorted;   // sorted copy of _k for distinct-value counting
    std::vector<float> _absErrors;
};
}  // namespace saint
