#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "Cepstrum.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetector.h"
#include "RealFft.h"
#include "Utils.h"

class PitchDetectorLoggerInterface;

namespace saint {
class AutocorrPitchDetector {
   public:
    AutocorrPitchDetector(int sampleRate, int fftSize, const std::vector<float>& fftWindow,
                          float minFreq, PitchDetectorLoggerInterface& logger);

    // Returns the in-range fundamental-frequency estimate (Hz), or 0 if no in-range pitch is
    // found, and writes its [0,1] presence score to *presenceScore.
    //
    // If doubleBandEstimate is non-null it additionally receives a second estimate (Hz) taken
    // from the ACF peak over an extended, lower band (down to lowBandMinFreqRatio * minFreq;
    // see enableLowBandSearch in PitchDetectorTypes.h), with its presence in
    // *doubleBandPresenceScore. When the global ACF peak lies in range the two estimates
    // coincide; when a stronger peak sits below the in-range floor (a string tuned up from
    // slack) doubleBandEstimate is lower than the returned estimate. The caller selects between
    // them; a double-band estimate must not be octave-disambiguated against the in-range tuning
    // (its fundamental is below minF0).
    float process(const std::vector<std::complex<float>>& dft, float* presenceScore,
                  std::optional<float> constraint = std::nullopt,
                  float* doubleBandEstimate = nullptr, float* doubleBandPresenceScore = nullptr);

    // Drop the cross-frame averaging history. Call on onset so a new note's
    // autocorrelation never blurs into the previous note's.
    void reset();

   private:
    // Push the latest autocorrelation frame into the ring buffer and return the
    // average over the last autocorrAveragingFrameCount frames (see idea #1 in
    // acf-denoising-ideas.md). Returns xcorr unchanged when averaging is disabled.
    const std::vector<float>& averageOverFrames(const std::vector<float>& xcorr);

    // Quad-interpolated frequency (Hz) for an ACF peak at lag `maxIndex`, or 0 when maxIndex is 0.
    float refineEstimate(const std::vector<float>& acf, int maxIndex) const;

    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    RealFft _fwdFft;
    const std::vector<float> _lpWindow;
    const int _lastSearchIndex;
    // Longest lag scanned by the double-band search (one octave below _lastSearchIndex, clamped
    // to the half-spectrum). Equals _lastSearchIndex when the fallback is disabled or the window
    // is already at its Nyquist-limited floor, which makes the double-band estimate coincide
    // with the in-range one.
    const int _lowBandLastSearchIndex;
    const std::vector<float> _windowXcorr;

    // Reused scratch, so process() allocates nothing on the audio thread: _xcorr holds
    // the (inverse-FFT) autocorrelation; _freqScratch a mutable copy of the input
    // spectrum that getXCorr overwrites in place.
    std::vector<float> _xcorr;
    std::vector<std::complex<float>> _freqScratch;

    // Cross-frame averaging state (idea #1). Empty/unused when averaging is off.
    std::vector<std::vector<float>> _xcorrHistory;
    std::vector<float> _averagedXcorr;
    int _historyWritePos = 0;
    int _historyFilled = 0;
};
}  // namespace saint
