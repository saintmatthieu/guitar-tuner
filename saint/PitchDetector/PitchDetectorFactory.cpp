#include "PitchDetectorFactory.h"

#include "AutocorrEstimateDisambiguator.h"
#include "DummyPitchDetectorLogger.h"
#include "FrequencyDomainTransformer.h"
#include "LowBandAnalyzer.h"
#include "OnsetDetector.h"
#include "PitchDetectionHolder.h"
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
                                                    LowBandConfig lowBand) {
    auto logger = std::make_unique<DummyPitchDetectorLogger>();

    const auto minFreq = getMinFreq(tuning);

    // The preprocessor low-passes at the full rate and then decimates by D; everything
    // downstream of it runs at fs/D with a ~D-times smaller FFT. The onset detector keeps
    // the full-rate broadband audio. Mirrors BenchmarkAlgorithms.cpp::createImpl.
    const auto decimatedSampleRate = sampleRate / defaultDecimationFactor;
    const auto decimatedBlockSize = samplesPerBlockPerChannel / defaultDecimationFactor;

    FrequencyDomainTransformer transformer(decimatedSampleRate, channelFormat, decimatedBlockSize,
                                           minFreq, *logger);

    AutocorrPitchDetector autocorrPitchDetector(decimatedSampleRate, transformer.fftSize(),
                                                transformer.window(), minFreq, *logger);

    AutocorrEstimateDisambiguator disambiguator(decimatedSampleRate, transformer.fftSize(), tuning,
                                                *logger);

    OnsetDetector onsetDetector(sampleRate, channelFormat, samplesPerBlockPerChannel, minFreq);

    auto preprocessor = std::make_unique<Preprocessor>(
        sampleRate, channelFormat, samplesPerBlockPerChannel, defaultDecimationFactor);

    // Below-range analysis (LowBandConfig): runs on the preprocessor's output, hence the
    // decimated rate and block size. Null while the feature is off, which costs nothing.
    auto lowBandAnalyzer = std::make_unique<LowBandAnalyzer>(
        decimatedSampleRate, channelFormat, decimatedBlockSize, minFreq, *logger, lowBand);

    auto impl = std::make_unique<PitchDetectorImpl>(
        std::move(preprocessor), std::move(transformer), std::move(autocorrPitchDetector),
        std::move(disambiguator), std::move(onsetDetector), std::move(lowBandAnalyzer),
        std::move(logger), defaultDecimationFactor, OctaviationGateConfig{}, lowBand);

    auto medianFilter = std::make_unique<PitchDetectorMedianFilter>(
        sampleRate, samplesPerBlockPerChannel, std::move(impl));

    const auto blocksPerSecond = sampleRate / samplesPerBlockPerChannel;

    auto holder =
        std::make_unique<PitchDetectionSmoother>(std::move(medianFilter), blocksPerSecond);

    return std::make_unique<PitchDetectionHolder>(std::move(holder), sampleRate,
                                                  samplesPerBlockPerChannel);
}
}  // namespace

std::unique_ptr<IssueReportingPitchDetector> PitchDetectorFactory::createInstance(
    int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel, Tuning tuning,
    std::function<void(std::string logLine)> cpuSummaryCallback, LowBandConfig lowBand) {
    const recording::PitchDetectorConfig config{sampleRate, channelFormat,
                                                samplesPerBlockPerChannel, tuning};
    return std::make_unique<IssueReportingPitchDetector>(
        config,
        [config, lowBand] {
            return createImplementation(config.sampleRate, config.channelFormat,
                                        config.samplesPerBlockPerChannel, config.tuning, lowBand);
        },
        std::move(cpuSummaryCallback));
}
}  // namespace saint
