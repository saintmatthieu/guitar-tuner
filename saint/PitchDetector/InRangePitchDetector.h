#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "FrequencyDomainTransformer.h"
#include "LowBandAnalyzer.h"
#include "OnsetDetector.h"
#include "PitchDetector.h"
#include "PitchDetectorLoggerInterface.h"
#include "Preprocessor.h"

namespace saint {
class InRangePitchDetector : public PitchDetector {
   public:
    InRangePitchDetector(std::unique_ptr<Preprocessor>, FrequencyDomainTransformer,
                         AutocorrPitchDetector, AutocorrEstimateDisambiguator, OnsetDetector,
                         std::unique_ptr<LowBandAnalyzer> lowBandAnalyzer,
                         std::unique_ptr<PitchDetectorLoggerInterface> logger, int decimationFactor,
                         OctaviationGateConfig gate = {}, LowBandConfig lowBand = {});

    PitchDetectionResult process(const float*, DebugOutput*,
                                 std::vector<float>* debugOutputSignal = nullptr);
    int delaySamples() const {
        return _decimationFactor * windowSizeSamples() / 2;
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
    bool likelyLowBand(DebugOutput*, float disambiguatedEstimate, bool logging);
    // Writes the below-range decision - the candidates weighed, the winning comb, the verdict
    // and the thresholds it was held to - to the logger, for eval/showLowBandAnalysis.py.
    void logLowBand(const LowBandAnalyzer::Verdict&, float inRangeEstimate) const;

    const std::unique_ptr<Preprocessor> _preprocessor;
    FrequencyDomainTransformer _frequencyDomainTransformer;
    AutocorrPitchDetector _autocorrPitchDetector;
    AutocorrEstimateDisambiguator _disambiguator;
    OnsetDetector _onsetDetector;
    const std::unique_ptr<LowBandAnalyzer> _lowBandAnalyzer;

    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    std::optional<float> _estimateConstraint;
    // Filled only on the frame the logger records; reused, so it stops allocating after that.
    LowBandAnalyzer::Diagnostics _lowBandDiagnostics;
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
    const LowBandConfig _lowBand;
    const int _decimationFactor;

    // Reused dB-power-spectrum buffer, so process() allocates nothing on the audio
    // thread. Sized on the first block, then reused in place.
    std::vector<float> _dbSpectrum;
};
}  // namespace saint
