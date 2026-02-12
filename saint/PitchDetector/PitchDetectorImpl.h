#pragma once

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetector.h"

class PitchDetectorLoggerInterface;

namespace saint {
class PitchDetectorImpl : public PitchDetector {
   public:
    PitchDetectorImpl(FrequencyDomainTransformer, AutocorrPitchDetector,
                      AutocorrEstimateDisambiguator,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);
    ~PitchDetectorImpl() override = default;

    float process(const float*, float* presenceScore) override;
    int delaySamples() const override {
        return windowSizeSamples() / 2;
    }

    void setEstimateConstraint(std::optional<float> constraint) override {
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
