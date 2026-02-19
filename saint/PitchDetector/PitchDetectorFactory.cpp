#include "PitchDetectorFactory.h"

#include "AutocorrEstimateDisambiguator.h"
#include "DummyPitchDetectorLogger.h"
#include "FrequencyDomainTransformer.h"
#include "OnsetDetector.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorMedianFilter.h"
#include "PitchDetectorUtils.h"
#include "Preprocessor.h"

namespace saint {

std::unique_ptr<PitchDetector> PitchDetectorFactory::createInstance(
    int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
    const std::optional<PitchDetectorConfig>& config) {
    auto logger = std::make_unique<DummyPitchDetectorLogger>();

    const auto minFreq = getMinFreq(config);

    FrequencyDomainTransformer transformer(sampleRate, channelFormat, samplesPerBlockPerChannel,
                                           minFreq, *logger);

    AutocorrPitchDetector autocorrPitchDetector(sampleRate, transformer.fftSize(),
                                                transformer.window(), minFreq, *logger);

    AutocorrEstimateDisambiguator disambiguator(sampleRate, transformer.fftSize(), config, *logger);

    OnsetDetector onsetDetector(sampleRate, channelFormat, samplesPerBlockPerChannel, minFreq);

    auto preprocessor =
        std::make_unique<Preprocessor>(sampleRate, channelFormat, samplesPerBlockPerChannel);

    auto impl = std::make_unique<PitchDetectorImpl>(
        std::move(preprocessor), std::move(transformer), std::move(autocorrPitchDetector),
        std::move(disambiguator), std::move(onsetDetector), std::move(logger));

    return std::make_unique<PitchDetectorMedianFilter>(sampleRate, samplesPerBlockPerChannel,
                                                       std::move(impl));
}
}  // namespace saint