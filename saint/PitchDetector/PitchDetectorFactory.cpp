#include "PitchDetectorFactory.h"

#include "AutocorrEstimateDisambiguator.h"
#include "DummyPitchDetectorLogger.h"
#include "FrequencyDomainTransformer.h"
#include "OnsetDetector.h"
#include "PitchDetectionSmoother.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorMedianFilter.h"
#include "PitchDetectorUtils.h"
#include "Preprocessor.h"

namespace saint {

std::unique_ptr<PitchDetector> PitchDetectorFactory::createInstance(
    int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel, Tuning tuning) {
    auto logger = std::make_unique<DummyPitchDetectorLogger>();

    const auto minFreq = getMinFreq(tuning);

    FrequencyDomainTransformer transformer(sampleRate, channelFormat, samplesPerBlockPerChannel,
                                           minFreq, *logger);

    AutocorrPitchDetector autocorrPitchDetector(sampleRate, transformer.fftSize(),
                                                transformer.window(), minFreq, *logger);

    AutocorrEstimateDisambiguator disambiguator(sampleRate, transformer.fftSize(), tuning, *logger);

    OnsetDetector onsetDetector(sampleRate, channelFormat, samplesPerBlockPerChannel, minFreq);

    auto preprocessor =
        std::make_unique<Preprocessor>(sampleRate, channelFormat, samplesPerBlockPerChannel);

    auto impl = std::make_unique<PitchDetectorImpl>(
        std::move(preprocessor), std::move(transformer), std::move(autocorrPitchDetector),
        std::move(disambiguator), std::move(onsetDetector), std::move(logger));

    auto medianFilter = std::make_unique<PitchDetectorMedianFilter>(
        sampleRate, samplesPerBlockPerChannel, std::move(impl));

    const auto blocksPerSecond = sampleRate / samplesPerBlockPerChannel;

    return std::make_unique<PitchDetectionSmoother>(std::move(medianFilter), blocksPerSecond);
}
}  // namespace saint