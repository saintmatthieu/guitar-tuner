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
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);

    float process(const float*, DebugOutput*);
    int delaySamples() const {
        return windowSizeSamples() / 2;
    }

    void setEstimateConstraint(float constraint) {
        _estimateConstraint = constraint;
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
};
}  // namespace saint
