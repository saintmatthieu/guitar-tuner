#include "PitchDetectorFactory.h"

#include "DummyPitchDetectorLogger.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorMedianFilter.h"

namespace saint {

std::unique_ptr<PitchDetector> PitchDetectorFactory::createInstance(
    int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
    const std::optional<PitchDetector::Config>& config) {
    auto impl =
        std::make_unique<PitchDetectorImpl>(sampleRate, channelFormat, samplesPerBlockPerChannel,
                                            config, std::make_unique<DummyPitchDetectorLogger>());
    return std::make_unique<PitchDetectorMedianFilter>(sampleRate, samplesPerBlockPerChannel,
                                                       std::move(impl));
}
}  // namespace saint