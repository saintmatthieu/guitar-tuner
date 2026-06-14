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

#if defined(SAINT_WITH_PESTO) || defined(SAINT_WITH_AUBIO)
#include <gtest/gtest.h>

#include "TestCaseUtils.h"
#endif

#ifdef SAINT_WITH_PESTO
#include "PestoPitchDetector.h"
#endif

#ifdef SAINT_WITH_AUBIO
#include "AubioPitchDetector.h"
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

#ifdef SAINT_WITH_AUBIO
// aubio exposes several pitch methods; each is registered as its own benchmark
// algorithm ("aubio-<method>") so runs are directly comparable.
const std::vector<std::string> kAubioMethods{"yin",   "yinfft", "yinfast", "mcomb",
                                             "fcomb", "schmitt", "specacf"};

std::unique_ptr<PitchDetector> createAubio(const std::string& method,
                                           const BenchmarkAlgorithmContext& ctx) {
    // Optional tuning knobs shared by all aubio methods (the wrapper picks
    // sensible defaults when these are absent):
    //   aubioBufSize=<n>     analysis window (power of two, >= blockSize)
    //   aubioConfidence=<f>  return 0 Hz below this confidence
    const auto argBufSize = getArgument<int>("aubioBufSize");
    const auto argConfidence = getArgument<std::string>("aubioConfidence");
    const auto bufSize = argBufSize.value_or(0);
    const auto confidence = argConfidence.has_value() ? std::stof(*argConfidence) : 0.f;
    return std::make_unique<AubioPitchDetector>(method, ctx.sampleRate, ctx.channelFormat,
                                                ctx.blockSize, bufSize, confidence);
}
#endif
}  // namespace

const std::map<std::string, BenchmarkAlgorithmFactory>& getBenchmarkAlgorithms() {
    static const std::map<std::string, BenchmarkAlgorithmFactory> algorithms = [] {
        std::map<std::string, BenchmarkAlgorithmFactory> map{
            {kDefaultAlgorithmId, createImpl},
#ifdef SAINT_WITH_PESTO
            {"pesto", createPesto},
#endif
        };
#ifdef SAINT_WITH_AUBIO
        for (const auto& method : kAubioMethods) {
            map["aubio-" + method] = [method](const BenchmarkAlgorithmContext& ctx) {
                return createAubio(method, ctx);
            };
        }
#endif
        return map;
    }();
    return algorithms;
}

}  // namespace saint
