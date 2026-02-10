#include "PitchDetectorFactory.h"

#include "DummyPitchDetectorLogger.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorMedianFilter.h"
#include "PitchDetectorUtils.h"

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

    auto impl = std::make_unique<PitchDetectorImpl>(std::move(transformer),
                                                    std::move(autocorrPitchDetector), sampleRate,
                                                    config, std::move(logger));

    return std::make_unique<PitchDetectorMedianFilter>(sampleRate, samplesPerBlockPerChannel,
                                                       std::move(impl));
}
}  // namespace saint