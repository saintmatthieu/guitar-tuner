#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "PitchDetector.h"
#include "PitchDetectorTypes.h"

namespace saint {

// Everything a benchmark algorithm factory may need to build a detector
// instance for one test case.
struct BenchmarkAlgorithmContext {
    int sampleRate = 0;
    ChannelFormat channelFormat = ChannelFormat::Mono;
    int blockSize = 0;
    Tuning tuning = Tuning::Standard;
    // In-house-specific options; other algorithms are free to ignore them.
    std::optional<int> indexOfProcessToLog;
    bool withMedianFilter = true;
    // When false, the probNotOctaviated gate is bypassed so every frame emits an
    // estimate. Used to collect the full presence-score/error distribution for
    // re-fitting the gate (see eval/fitAndShowErrorProbabilityModels.py).
    bool applyOctaviationGate = true;
    // Gate knobs (in-house algorithm). presenceThreshold cuts the fitted
    // probNotOctaviated; harmonicityFloor (0 = off) rejects estimates lacking
    // harmonic support. Overridable from the CLI to sweep operating points.
    double presenceThreshold = 0.85;
    float harmonicityFloor = 0.f;
    // Median-filter window (s). Drives output latency; overridable to test reverting
    // the 0.2 s bump once other safeguards (e.g. the harmonicity floor) carry the load.
    float medianFilterDuration = 0.2f;
};

using BenchmarkAlgorithmFactory =
    std::function<std::unique_ptr<PitchDetector>(const BenchmarkAlgorithmContext&)>;

// Aggregate metrics the benchmark computes over a full-corpus run. These are the
// inputs to an algorithm's pass/fail gates.
struct BenchmarkMetrics {
    double avgError = 0.;        // mean signed cents error
    double rmsError = 0.;        // mean of per-case RMS cents error
    double medianRmsError = 0.;  // median of per-case RMS cents error (robust central tendency)
    double p99RmsError = 0.;     // 99th-percentile per-case RMS cents error (tail accuracy)
    double falsePositiveRate = 0.;
    double falseNegativeRate = 0.;  // weighted
    double auc = 0.;                // area under the presence-score ROC curve
};

// One metric an algorithm is gated on. The reference value is deliberately not
// stored here: it lives in a golden file (BenchmarkingOutput/<fileStem><suffix>.txt),
// seeded on the first run and compared within `tolerance` on subsequent runs.
// This lets every algorithm - not just the in-house one - carry its own pass/fail
// criteria without baking reference numbers into the test.
struct MetricGate {
    std::string displayName;                               // shown in the pass/fail message
    std::string fileStem;                                  // golden-file basename
    std::function<double(const BenchmarkMetrics&)> value;  // which metric to read
    double tolerance = 0.01;
};

// An algorithm available for benchmarking: how to build it, and which metrics
// decide its pass/fail. An empty `gates` list means the metrics are reported but
// not gated.
struct BenchmarkAlgorithm {
    BenchmarkAlgorithmFactory create;
    std::vector<MetricGate> gates;
};

// The default in-house algorithm's ID.
extern const std::string kDefaultAlgorithmId;

// All algorithms available for benchmarking, keyed by algorithm ID.
const std::map<std::string, BenchmarkAlgorithm>& getBenchmarkAlgorithms();

}  // namespace saint
