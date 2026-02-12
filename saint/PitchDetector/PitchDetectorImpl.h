#pragma once

#include <memory>
#include <optional>

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
class PitchDetectorImpl {
   public:
    PitchDetectorImpl(FrequencyDomainTransformer, AutocorrPitchDetector,
                      AutocorrEstimateDisambiguator,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);

    float process(const float*, float* presenceScore);
    int delaySamples() const {
        return windowSizeSamples() / 2;
    }

    void setEstimateConstraint(std::optional<float> constraint) {
        _estimateConstraint = constraint;
    }

    int windowSizeSamples() const {
        return _frequencyDomainTransformer.windowSizeSamples();
    }

   private:
    FrequencyDomainTransformer _frequencyDomainTransformer;
    AutocorrPitchDetector _autocorrPitchDetector;
    AutocorrEstimateDisambiguator _disambiguator;

    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    std::optional<float> _estimateConstraint;
};
}  // namespace saint
