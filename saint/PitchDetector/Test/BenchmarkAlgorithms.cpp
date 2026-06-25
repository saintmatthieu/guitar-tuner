#include "BenchmarkAlgorithms.h"

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "DummyPitchDetectorLogger.h"
#include "FrequencyDomainTransformer.h"
#include "OnsetDetector.h"
#include "PitchDetectionHolder.h"
#include "PitchDetectionSmoother.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorImplTestWrapper.h"
#include "PitchDetectorLogger.h"
#include "PitchDetectorMedianFilter.h"
#include "PitchDetectorUtils.h"
#include "Preprocessor.h"

#if defined(SAINT_WITH_PESTO) || defined(SAINT_WITH_AUBIO) || defined(SAINT_WITH_PYIN)
#include <gtest/gtest.h>

#include "TestCaseUtils.h"
#endif

#ifdef SAINT_WITH_PESTO
#include "PestoPitchDetector.h"
#endif

#ifdef SAINT_WITH_AUBIO
#include "AubioPitchDetector.h"
#endif

#ifdef SAINT_WITH_PYIN
#include "PyinPitchDetector.h"
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

    // The preprocessor low-passes at the full rate and then decimates by D; everything
    // downstream of it runs at the decimated rate fsD with a ~D-times smaller FFT (the
    // CPU win). The onset detector keeps the full-rate broadband audio. fsD and the
    // decimated block size are exact for the corpus's 44.1/48 kHz at D in {1,2,3,4}
    // (the per-block count still rounds for 44.1 kHz at D=2/4, which the transformer
    // tolerates).
    const auto D = ctx.decimationFactor;
    const auto decimatedSampleRate = ctx.sampleRate / D;
    const auto decimatedBlockSize = ctx.blockSize / D;

    auto preprocessor =
        std::make_unique<Preprocessor>(ctx.sampleRate, ctx.channelFormat, ctx.blockSize, D);

    FrequencyDomainTransformer transformer(decimatedSampleRate, ctx.channelFormat,
                                           decimatedBlockSize, minFreq, *logger);
    AutocorrPitchDetector autocorrPitchDetector(decimatedSampleRate, transformer.fftSize(),
                                                transformer.window(), minFreq, *logger);
    AutocorrEstimateDisambiguator disambiguator(decimatedSampleRate, transformer.fftSize(),
                                                ctx.tuning, *logger);
    OnsetDetector onsetDetector(ctx.sampleRate, ctx.channelFormat, ctx.blockSize, minFreq,
                                ctx.onset);

    auto internalAlgorithm = std::make_unique<PitchDetectorImpl>(
        std::move(preprocessor), std::move(transformer), std::move(autocorrPitchDetector),
        std::move(disambiguator), std::move(onsetDetector), std::move(logger), D, ctx.gate);

    if (!ctx.withMedianFilter) {
        return std::make_unique<PitchDetectorImplTestWrapper>(std::move(internalAlgorithm));
    }

    auto medianFilter = std::make_unique<PitchDetectorMedianFilter>(
        ctx.sampleRate, ctx.blockSize, std::move(internalAlgorithm), ctx.medianFilter);
    auto holder = std::make_unique<PitchDetectionHolder>(std::move(medianFilter), ctx.sampleRate,
                                                         ctx.blockSize, ctx.hold);
    const auto blocksPerSecond = ctx.sampleRate / ctx.blockSize;
    return std::make_unique<PitchDetectionSmoother>(std::move(holder), blocksPerSecond);
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
const std::vector<std::string> kAubioMethods{"yin",   "yinfft",  "yinfast", "mcomb",
                                             "fcomb", "schmitt", "specacf"};

std::unique_ptr<PitchDetector> createAubio(const std::string& method,
                                           const BenchmarkAlgorithmContext& ctx) {
    // Optional tuning knobs shared by all aubio methods (the wrapper picks
    // sensible defaults when these are absent):
    //   aubioBufSize=<n>     analysis window (power of two, >= blockSize)
    //   aubioConfidence=<f>  return 0 Hz below this confidence (overrides the
    //                        method's built-in 1%-FPR default)
    const auto argBufSize = getArgument<int>("aubioBufSize");
    const auto argConfidence = getArgument<std::string>("aubioConfidence");
    const auto bufSize = argBufSize.value_or(0);
    const auto confidence = argConfidence.has_value() ? std::stof(*argConfidence) : -1.f;
    return std::make_unique<AubioPitchDetector>(method, ctx.sampleRate, ctx.channelFormat,
                                                ctx.blockSize, bufSize, confidence);
}
#endif

#ifdef SAINT_WITH_PYIN
std::unique_ptr<PitchDetector> createPyin(const BenchmarkAlgorithmContext& ctx) {
    // pYIN does its own (HMM) temporal smoothing, so — like the other third-party
    // algorithms — it is benchmarked raw, without the in-house median filter and
    // smoother. Optional tuning knobs (the wrapper picks pYIN's reference defaults
    // when these are absent):
    //   pyinFrameSize=<n>      analysis window, power of two (default 2048)
    //   pyinFixedLag=<n>       HMM smoothing lag in frames (latency = (n-1) blocks)
    //   pyinThreshDistr=<0..4> YIN threshold-distribution prior (2 = Beta mean .15)
    //   pyinLowAmp=<f>         block-RMS below which candidates are suppressed (default .1)
    const auto frameSize = getArgument<int>("pyinFrameSize").value_or(2048);
    const auto fixedLag = getArgument<int>("pyinFixedLag").value_or(20);
    const auto threshDistr = getArgument<int>("pyinThreshDistr").value_or(2);
    const auto argLowAmp = getArgument<std::string>("pyinLowAmp");
    const auto lowAmp = argLowAmp.has_value() ? std::stof(*argLowAmp) : 0.1f;
    return std::make_unique<PyinPitchDetector>(ctx.sampleRate, ctx.channelFormat, ctx.blockSize,
                                               frameSize, fixedLag, threshDistr, lowAmp);
}
#endif

// Gate builders, shared across algorithms. Each maps to a golden file basename
// (BenchmarkingOutput/<stem>[_<algorithm>].txt) holding the reference value.
MetricGate rmsGate() {
    return {"RMS error", "RMS_error", [](const BenchmarkMetrics& m) { return m.rmsError; }, 0.01};
}
// Median per-case RMS: robust central-tendency accuracy, insensitive to how large
// the catastrophic (octave-class) errors happen to be.
MetricGate medianRmsGate() {
    return {"median RMS error", "median_RMS_error",
            [](const BenchmarkMetrics& m) { return m.medianRmsError; }, 0.01};
}
// 99th-percentile per-case RMS: a continuous tail metric. It sits right at the
// gross-error boundary (~57c on master), so it tracks how often catastrophic
// (octave/competing-pitch) errors occur without conflating that with their
// magnitude the way mean RMS does, and with finer granularity than a count.
MetricGate p99RmsGate() {
    return {"99th-pct RMS error", "p99_RMS_error",
            [](const BenchmarkMetrics& m) { return m.p99RmsError; }, 0.01};
}
MetricGate fnrGate() {
    return {"weighted FNR", "FNR", [](const BenchmarkMetrics& m) { return m.falseNegativeRate; },
            0.01};
}
MetricGate aucGate() {
    return {"AUC", "AUC", [](const BenchmarkMetrics& m) { return m.auc; }, 0.01};
}
}  // namespace

const std::map<std::string, BenchmarkAlgorithm>& getBenchmarkAlgorithms() {
    static const std::map<std::string, BenchmarkAlgorithm> algorithms = [] {
        std::map<std::string, BenchmarkAlgorithm> map;
        // In-house: gated on median RMS cents error, 99th-percentile RMS cents error,
        // weighted FNR and presence-score AUC. (Mean RMS is still reported but not
        // gated: it conflates catastrophic-error frequency with their magnitude.)
        map[kDefaultAlgorithmId] = {createImpl,
                                    {medianRmsGate(), p99RmsGate(), fnrGate(), aucGate()}};
#ifdef SAINT_WITH_PESTO
        // PESTO: only RMS error and FNR are gated; its confidence calibration is a
        // separate concern, so its AUC is reported but not verified.
        map["pesto"] = {createPesto, {rmsGate(), fnrGate()}};
#endif
#ifdef SAINT_WITH_AUBIO
        // aubio: gated like the in-house algorithm, including the presence-score AUC.
        for (const auto& method : kAubioMethods) {
            map["aubio-" + method] = {
                [method](const BenchmarkAlgorithmContext& ctx) { return createAubio(method, ctx); },
                {rmsGate(), fnrGate(), aucGate()}};
        }
#endif
#ifdef SAINT_WITH_PYIN
        // pYIN: gated like the in-house algorithm, including the presence-score AUC.
        map["pyin"] = {createPyin, {rmsGate(), fnrGate(), aucGate()}};
#endif
        return map;
    }();
    return algorithms;
}

}  // namespace saint
