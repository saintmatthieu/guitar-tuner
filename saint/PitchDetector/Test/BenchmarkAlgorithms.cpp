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

#ifdef SAINT_WITH_PESTO
#include <gtest/gtest.h>

#include "PestoPitchDetector.h"
#include "TestCaseUtils.h"
#endif

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

#ifdef SAINT_WITH_PESTO
std::unique_ptr<PitchDetector> createPesto(const BenchmarkAlgorithmContext& ctx) {
    // Models are exported with realtime.export_onnx, which freezes sample rate
    // and chunk size and encodes them in the filename.
    const auto modelPath = std::filesystem::path(SAINT_PESTO_MODEL_DIR) /
                           ("mir-1k_g7_" + std::to_string(ctx.sampleRate) + "_" +
                            std::to_string(ctx.blockSize) + ".onnx");
    const auto argThreshold = getArgument<std::string>("pestoThreshold");
    const auto threshold = argThreshold.has_value() ? std::stof(*argThreshold) : 0.5f;
    return std::make_unique<PestoPitchDetector>(modelPath, ctx.sampleRate, ctx.channelFormat,
                                                ctx.blockSize, threshold);
}
#endif
}  // namespace

const std::map<std::string, BenchmarkAlgorithmFactory>& getBenchmarkAlgorithms() {
    static const std::map<std::string, BenchmarkAlgorithmFactory> algorithms{
        {kDefaultAlgorithmId, createImpl},
#ifdef SAINT_WITH_PESTO
        {"pesto", createPesto},
#endif
    };
    return algorithms;
}

}  // namespace saint
