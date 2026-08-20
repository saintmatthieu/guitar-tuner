#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
class InRangePitchDetector {
   public:
    InRangePitchDetector(FrequencyDomainTransformer, AutocorrPitchDetector,
                         AutocorrEstimateDisambiguator, PitchDetectorLoggerInterface& logger,
                         OctaviationGateConfig gate = {});

    // `audio` is the preprocessor's output, so the rate is the decimated one.
    PitchDetectionResult process(const std::vector<float>& audio, DebugOutput*);

    // A new attack, likely a new note: drop the lock and the cross-frame autocorrelation
    // average so the new note doesn't blur into the previous one.
    void onNewOnset();

    int delaySamples() const {
        return windowSizeSamples() / 2;
    }

    std::pair<float, float> pitchSearchRange() const {
        return _disambiguator.searchRange();
    }

    void setEstimateConstraint(float constraint) {
        _estimateConstraint = constraint;
    }

    int windowSizeSamples() const {
        return _frequencyDomainTransformer.windowSizeSamples();
    }

   private:
    FrequencyDomainTransformer _frequencyDomainTransformer;
    AutocorrPitchDetector _autocorrPitchDetector;
    AutocorrEstimateDisambiguator _disambiguator;
    PitchDetectorLoggerInterface& _logger;
    std::optional<float> _estimateConstraint;
    // When false, the probNotOctaviated gate is bypassed so every frame emits its
    // estimate. Used to collect the full presence-score/error distribution for
    // re-fitting the gate's Bayesian model (see eval/fitAndShowErrorProbabilityModels.py).
    const bool _applyOctaviationGate;
    // Gate thresholds. _presenceThreshold (no-constraint) is the cut on the fitted
    // probNotOctaviated; _harmonicityFloor rejects estimates lacking harmonic support
    // (#4). Floor 0 disables the harmonicity criterion (legacy behaviour).
    // _presenceThresholdWithConstraint replaces _presenceThreshold once a constraint is
    // locked (phase 2 / tracking); see octaviationPresenceThresholdWithConstraint.
    const double _presenceThreshold;
    const float _harmonicityFloor;
    const double _presenceThresholdWithConstraint;

    // Reused dB-power-spectrum buffer, so process() allocates nothing on the audio
    // thread. Sized on the first block, then reused in place.
    std::vector<float> _dbSpectrum;
};
}  // namespace saint
