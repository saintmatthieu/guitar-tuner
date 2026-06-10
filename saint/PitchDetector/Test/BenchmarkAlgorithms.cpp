#include "BenchmarkAlgorithms.h"

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "DummyPitchDetectorLogger.h"
#include "FrequencyDomainTransformer.h"
#include "OnsetDetector.h"
#include "PitchDetectionSmoother.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorImplTestWrapper.h"
#include "PitchDetectorLogger.h"
#include "PitchDetectorMedianFilter.h"
#include "PitchDetectorUtils.h"
#include "Preprocessor.h"

namespace saint {

const std::string kDefaultAlgorithmId = "impl";

namespace {
std::unique_ptr<PitchDetector> createImpl(const BenchmarkAlgorithmContext& ctx) {
    std::unique_ptr<PitchDetectorLoggerInterface> logger;
    if (ctx.indexOfProcessToLog.has_value()) {
        logger = std::make_unique<PitchDetectorLogger>(ctx.sampleRate, *ctx.indexOfProcessToLog);
    } else {
        logger = std::make_unique<DummyPitchDetectorLogger>();
    }

    const auto minFreq = getMinFreq(ctx.tuning);
    auto preprocessor =
        std::make_unique<Preprocessor>(ctx.sampleRate, ctx.channelFormat, ctx.blockSize);

    FrequencyDomainTransformer transformer(ctx.sampleRate, ctx.channelFormat, ctx.blockSize,
                                           minFreq, *logger);
    AutocorrPitchDetector autocorrPitchDetector(ctx.sampleRate, transformer.fftSize(),
                                                transformer.window(), minFreq, *logger);
    AutocorrEstimateDisambiguator disambiguator(ctx.sampleRate, transformer.fftSize(), ctx.tuning,
                                                *logger);
    OnsetDetector onsetDetector(ctx.sampleRate, ctx.channelFormat, ctx.blockSize, minFreq);

    auto internalAlgorithm = std::make_unique<PitchDetectorImpl>(
        std::move(preprocessor), std::move(transformer), std::move(autocorrPitchDetector),
        std::move(disambiguator), std::move(onsetDetector), std::move(logger));

    if (!ctx.withMedianFilter) {
        return std::make_unique<PitchDetectorImplTestWrapper>(std::move(internalAlgorithm));
    }

    auto medianFilter = std::make_unique<PitchDetectorMedianFilter>(ctx.sampleRate, ctx.blockSize,
                                                                    std::move(internalAlgorithm));
    const auto blocksPerSecond = ctx.sampleRate / ctx.blockSize;
    return std::make_unique<PitchDetectionSmoother>(std::move(medianFilter), blocksPerSecond);
}
}  // namespace

const std::map<std::string, BenchmarkAlgorithmFactory>& getBenchmarkAlgorithms() {
    static const std::map<std::string, BenchmarkAlgorithmFactory> algorithms{
        {kDefaultAlgorithmId, createImpl},
    };
    return algorithms;
}

}  // namespace saint
