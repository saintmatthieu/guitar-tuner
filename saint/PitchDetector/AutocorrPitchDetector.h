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
                          float minFreq, PitchDetectorLoggerInterface& logger,
                          bool applyConstraintBandPass = false);

    // `harmonicConsistencyCents` (optional out): how far the autocorrelation peak near twice the
    // fundamental lag sits from exactly 2x the refined fundamental lag, in cents. ~0 for a clean
    // periodic signal at any pitch (so a genuine pitch shift leaves it ~0), but grows when noise or
    // sub-fundamental contamination pulls the fundamental peak without moving the 2L peak with it -
    // i.e. a locked-phase reliability measure that is invariant to true shifts. `secondaryPeakScore`
    // (optional out): the normalised height of that 2L peak (its presence-score analogue); a near-
    // zero value means the harmonic structure has dissolved into noise.
    float process(const std::vector<std::complex<float>>& dft, float* presenceScore,
                  std::optional<float> constraint = std::nullopt,
                  float* harmonicConsistencyCents = nullptr, float* secondaryPeakScore = nullptr);

    // Drop the cross-frame averaging history. Call on onset so a new note's
    // autocorrelation never blurs into the previous note's.
    void reset();

   private:
    // Push the latest autocorrelation frame into the ring buffer and return the
    // average over the last autocorrAveragingFrameCount frames (see idea #1 in
    // acf-denoising-ideas.md). Returns xcorr unchanged when averaging is disabled.
    const std::vector<float>& averageOverFrames(const std::vector<float>& xcorr);

    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    // When false, the constraint band-pass is never applied (broadband ACF); see
    // autocorrConstraintBandHalfWidthSemitones. Off by default (production); the TestApp turns it
    // on so it can be auditioned live.
    const bool _applyConstraintBandPass;
    RealFft _fwdFft;
    const std::vector<float> _lpWindow;
    const int _lastSearchIndex;
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
