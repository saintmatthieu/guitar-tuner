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

// The benchmark disables the hold (PitchDetectionHolder) by default so the golden references
// measure the detector's intrinsic accuracy/FNR rather than the hold's note-persistence
// behaviour (production enables it via defaultHoldDuration in PitchDetectorTypes.h). Override:
// holdDuration=<s>.
constexpr float benchmarkHoldDuration = 0.0f;

// Everything a benchmark algorithm factory may need to build a detector
// instance for one test case.
struct BenchmarkAlgorithmContext {
    int sampleRate = 0;
    ChannelFormat channelFormat = ChannelFormat::Mono;
    int blockSize = 0;
    Tuning tuning = Tuning::Standard;
    // In-house preprocessor decimation factor (1 = none). The frequency-domain stages
    // then run at sampleRate/decimationFactor; see Preprocessor and createImpl.
    int decimationFactor = defaultDecimationFactor;
    // Lower bound of the pitch-search range, in semitones from the tuning's lowest open-string
    // note (negative = below; see getMinFreq). Default is the production operating point; the
    // CLI sweeps it via minFreqSemitoneOffset.
    int minFreqSemitoneOffset = defaultMinFreqSemitoneOffset;
    // In-house-specific options; other algorithms are free to ignore them.
    std::optional<int> indexOfProcessToLog;
    bool withMedianFilter = true;
    // In-house gate-tuning knobs (see PitchDetectorTypes.h). Each config defaults to the tuned
    // production operating point so a plain benchmark runs exactly what ships; the CLI overrides
    // individual fields to sweep operating points (see PitchDetectorImplTests.cpp). The one
    // deliberate deviation from production is that the benchmark disables the hold
    // (hold.holdDuration is set there): see benchmarkHoldDuration.
    OctaviationGateConfig gate;
    OnsetDetectorConfig onset;
    MedianFilterConfig medianFilter;
    HoldConfig hold;
    LowBandConfig lowBand;
};

using BenchmarkAlgorithmFactory =
    std::function<std::unique_ptr<PitchDetector>(const BenchmarkAlgorithmContext&)>;

// Aggregate metrics the benchmark computes over a full-corpus run. These are the
// inputs to an algorithm's pass/fail gates.
//
// The cents-error metrics only sample test cases whose true frequency lies within the
// detector's pitchSearchRange(): an out-of-range note (e.g. A1 with a standard tuning)
// is not expected to get a precise estimate, so its error is meaningless. Such cases
// still count towards FPR/FNR/AUC and, like every case, towards the bucket error rate,
// which scores the returned PitchBucket against the expected one (the truth frequency's
// bucket while the note sounds, nullopt outside).
struct BenchmarkMetrics {
    double avgError = 0.;        // mean signed cents error
    double rmsError = 0.;        // mean of per-case RMS cents error
    double medianRmsError = 0.;  // median of per-case RMS cents error (robust central tendency)
    double p99RmsError = 0.;     // 99th-percentile per-case RMS cents error (tail accuracy)
    double falsePositiveRate = 0.;
    double falseNegativeRate = 0.;  // weighted
    double auc = 0.;                // area under the presence-score ROC curve
    // Weighted rate of blocks whose PitchBucket differs from the expected one. Blocks
    // within the note carry the same decaying weight as the FNR (missing the bucket
    // while the note is loud counts more); blocks outside it carry weight 1.
    double bucketErrorRate = 0.;
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
