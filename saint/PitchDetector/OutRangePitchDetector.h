#pragma once

#include <memory>

#include "InRangePitchDetector.h"
#include "LowBandAnalyzer.h"
#include "OnsetDetector.h"
#include "PitchDetectorTypes.h"
#include "Preprocessor.h"

namespace saint {
/**
 * @brief Decides what a frame the in-range detector reports nothing for really is: silence, or a
 * string sounding outside the tuning's range.
 *
 * It owns the front end the two analyses share - the preprocessor, the onset detector and the
 * low-band analyzer - which is what lets it order them: the analyzer is fed the block before
 * anything asks it about that block.
 */
class OutRangePitchDetector : public PitchDetector {
   public:
    OutRangePitchDetector(std::unique_ptr<Preprocessor>, OnsetDetector,
                          std::unique_ptr<LowBandAnalyzer>, std::unique_ptr<InRangePitchDetector>,
                          std::unique_ptr<PitchDetectorLoggerInterface> logger,
                          int decimationFactor, LowBandConfig lowBand = {},
                          OutRangeConfig outRange = {});

    PitchDetectionResult process(const float* input, DebugOutput* = nullptr,
                                 std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;
    std::pair<float, float> pitchSearchRange() const override;

    void setEstimateConstraint(float constraint);

   private:
    bool likelyLowBand(DebugOutput*, float inRangeEstimate, bool logging);
    // Writes the below-range decision - the candidates weighed, the winning comb, the verdict
    // and the thresholds it was held to - to the logger, for eval/showLowBandAnalysis.py.
    void logLowBand(const LowBandAnalyzer::Verdict&, float inRangeEstimate) const;

    // Declared first, so it outlives everything below that logs through it.
    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    const std::unique_ptr<Preprocessor> _preprocessor;
    OnsetDetector _onsetDetector;
    const std::unique_ptr<LowBandAnalyzer> _lowBandAnalyzer;
    const std::unique_ptr<InRangePitchDetector> _inRangeDetector;
    const int _decimationFactor;
    const LowBandConfig _lowBand;
    const float _presenceThreshold;
    // Filled only on the frame the logger records; reused, so it stops allocating after that.
    LowBandAnalyzer::Diagnostics _lowBandDiagnostics;
    DebugOutput _debugOutput;
};
}  // namespace saint
