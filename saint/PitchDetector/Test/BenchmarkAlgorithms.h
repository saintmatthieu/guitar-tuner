#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

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
};

using BenchmarkAlgorithmFactory =
    std::function<std::unique_ptr<PitchDetector>(const BenchmarkAlgorithmContext&)>;

// The in-house algorithm, whose metrics are gated against the stored reference
// values in eval/BenchmarkingOutput.
extern const std::string kDefaultAlgorithmId;

// All algorithms available for benchmarking, keyed by algorithm ID.
const std::map<std::string, BenchmarkAlgorithmFactory>& getBenchmarkAlgorithms();

}  // namespace saint
