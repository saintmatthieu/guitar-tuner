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
    // noiseCompensation: enable noise-compensated autocorrelation (spectral
    // subtraction of a pre-roll noise estimate). Off by default: it improves
    // recall (FNR) and classifier AUC, but shifts the presence-score
    // distribution, so the Bayesian gate should be re-fitted before it is
    // turned on in production.
    AutocorrPitchDetector(int sampleRate, int fftSize, const std::vector<float>& fftWindow,
                          float minFreq, PitchDetectorLoggerInterface& logger,
                          bool noiseCompensation = false);

    // If noise compensation is active for this frame and `compensatedSpectrum`
    // is non-null, the noise-subtracted spectrum used for the autocorrelation is
    // copied out so that downstream stages (e.g. the octave disambiguator) can
    // work on the same denoised data. It is left untouched (empty) otherwise.
    float process(const std::vector<std::complex<float>>& dft, float* presenceScore,
                  std::optional<float> constraint = std::nullopt,
                  std::vector<std::complex<float>>* compensatedSpectrum = nullptr);

    // Stop adapting the noise-power estimate. Called once a note is active (on
    // an onset): the presence score is an unreliable noise/speech discriminator
    // under noise, so we only ever learn the noise floor from the reliably
    // note-free audio that precedes the first onset, then hold it fixed.
    void freezeNoiseEstimation() {
        _noiseEstimationFrozen = true;
    }

   private:
    // Attenuate each spectral bin so that |X|^2 becomes the noise-subtracted
    // power. Modifies `spectrum` in place.
    void applyNoiseCompensation(std::vector<std::complex<float>>& spectrum) const;
    // Update the per-bin noise-power estimate from this frame's spectrum, gated
    // by the presence score so frames containing a pitch don't pollute it.
    void updateNoiseEstimate(const std::vector<std::complex<float>>& spectrum,
                             float presenceScore);

    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    RealFft _fwdFft;
    const std::vector<float> _lpWindow;
    const int _lastSearchIndex;
    const std::vector<float> _windowXcorr;
    const bool _noiseCompensation;
    // Number of low-pass bins (lpWindow > 0) over which the noise floor is
    // tracked and subtracted. These are the bins that affect the
    // autocorrelation; the high bins are left intact so the disambiguator can
    // still estimate its own noise floor from them.
    const int _noiseBinCount;
    std::vector<float> _noisePsd;
    int _noiseFramesSeen = 0;
    bool _noiseEstimationFrozen = false;
};
}  // namespace saint
