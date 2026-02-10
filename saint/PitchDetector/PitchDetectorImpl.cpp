#include "PitchDetectorImpl.h"

#include "PitchDetectorLoggerInterface.h"

namespace saint {
PitchDetectorImpl::PitchDetectorImpl(FrequencyDomainTransformer transformer,
                                     AutocorrPitchDetector autocorrPitchDetector,
                                     AutocorrEstimateDisambiguator disambiguator,
                                     std::unique_ptr<PitchDetectorLoggerInterface> logger)
    : _frequencyDomainTransformer(std::move(transformer)),
      _autocorrPitchDetector(std::move(autocorrPitchDetector)),
      _disambiguator(std::move(disambiguator)),
      _logger(std::move(logger)) {}

float PitchDetectorImpl::process(const float* audio, float* presenceScore) {
    _logger->StartNewEstimate();
    utils::Finally finally{[this] { _logger->EndNewEstimate(nullptr, 0); }};

    const std::vector<std::complex<float>> freq = _frequencyDomainTransformer.process(audio);

    const float xcorrEstimate = _autocorrPitchDetector.process(freq, presenceScore);
    if (xcorrEstimate == 0.f) {
        return xcorrEstimate;
    }

    const auto disambiguatedEstimate = _disambiguator.process(xcorrEstimate, freq);

    return disambiguatedEstimate;
}
}  // namespace saint
