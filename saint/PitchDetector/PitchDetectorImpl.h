#pragma once

#include <memory>
#include <optional>

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "FrequencyDomainTransformer.h"
#include "OnsetDetector.h"
#include "PitchDetectorLoggerInterface.h"
#include "Preprocessor.h"

namespace saint {
class PitchDetectorImpl {
   public:
    PitchDetectorImpl(std::unique_ptr<Preprocessor>, FrequencyDomainTransformer,
                      AutocorrPitchDetector, AutocorrEstimateDisambiguator, OnsetDetector,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger,
                      OctaviationGateConfig gate = {});

    float process(const float*, DebugOutput*, std::vector<float>* debugOutputSignal = nullptr);
    int delaySamples() const {
        return windowSizeSamples() / 2;
    }

    void setEstimateConstraint(float constraint) {
        _estimateConstraint = constraint;
    }
    void clearEstimateConstraint() {
        _estimateConstraint.reset();
    }

    int windowSizeSamples() const {
        return _frequencyDomainTransformer.windowSizeSamples();
    }

   private:
    const std::unique_ptr<Preprocessor> _preprocessor;
    FrequencyDomainTransformer _frequencyDomainTransformer;
    AutocorrPitchDetector _autocorrPitchDetector;
    AutocorrEstimateDisambiguator _disambiguator;
    OnsetDetector _onsetDetector;

    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
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
};
}  // namespace saint
