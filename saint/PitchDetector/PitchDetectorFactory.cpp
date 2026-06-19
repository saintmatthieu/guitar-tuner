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
#include "Recording/IssueReportingPitchDetector.h"

namespace saint {

namespace {
std::unique_ptr<PitchDetector> createImplementation(int sampleRate, ChannelFormat channelFormat,
                                                    int samplesPerBlockPerChannel, Tuning tuning,
                                                    bool holdPitch) {
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

    // PitchDetectorConfig::holdPitch toggles the median filter's pitch-hold. Enabled, it uses
    // the filter's default hold window; disabled, a zero-length hold window turns it off. (The
    // default analysis window, 0.15 s, is restated for the disabled case so only the hold
    // window is overridden.)
    auto medianFilter =
        holdPitch ? std::make_unique<PitchDetectorMedianFilter>(
                        sampleRate, samplesPerBlockPerChannel, std::move(impl))
                  : std::make_unique<PitchDetectorMedianFilter>(
                        sampleRate, samplesPerBlockPerChannel, std::move(impl), 0.15f, 0.f);

    const auto blocksPerSecond = sampleRate / samplesPerBlockPerChannel;

    return std::make_unique<PitchDetectionSmoother>(std::move(medianFilter), blocksPerSecond);
}
}  // namespace

std::unique_ptr<IssueReportingPitchDetector> PitchDetectorFactory::createInstance(
    int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel, Tuning tuning,
    bool holdPitch, std::function<void(std::string logLine)> cpuSummaryCallback) {
    const recording::PitchDetectorConfig config{sampleRate, channelFormat,
                                                samplesPerBlockPerChannel, tuning, holdPitch};
    return std::make_unique<IssueReportingPitchDetector>(
        config,
        [config] {
            return createImplementation(config.sampleRate, config.channelFormat,
                                        config.samplesPerBlockPerChannel, config.tuning,
                                        config.holdPitch);
        },
        std::move(cpuSummaryCallback));
}
}  // namespace saint
