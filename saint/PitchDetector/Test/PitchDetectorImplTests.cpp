#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <thread>

#include "BenchmarkAlgorithms.h"
#include "TestCaseUtils.h"
#include "testUtils.h"

namespace saint {
namespace fs = std::filesystem;
namespace {
struct TestResult {
    std::string id;
    std::vector<testUtils::ProcessEstimate> estimates;
    // Cents-error stats; nullopt when the case's true frequency lies outside the
    // detector's pitchSearchRange() (no precise estimate expected there), so the case
    // never samples the pitch-error metrics.
    std::optional<testUtils::Cents> cents;
    double positiveWeight = 0.;
    int negativeCount = 0;
    int falsePositiveCount = 0;
    double falseNegativeWeight = 0.;
    // Bucket-mismatch tally for the bucket error rate (see BenchmarkMetrics): blocks
    // within the note carry the FNR decay weight, blocks outside it weight 1.
    double bucketErrorWeight = 0.;
    double bucketTotalWeight = 0.;
    double FPR = 0.;
    double FNR = 0.;
    std::string csvLine;
    fs::path testFile;
    fs::path noiseFile;
    std::string noiseRmsDb;
    double processCpuSeconds = 0.;  // summed CPU time spent in process() (Release only)
    double audioSeconds = 0.;       // audio duration fed through process()
};

struct ReferenceCheck {
    bool passed = false;
    bool seeded = false;  // golden file was just created (or refreshed)
    double reference = 0.;
};

// Compares an algorithm metric against its golden reference file. The file is
// seeded with the current value when it does not yet exist (or when `update` is
// set). On a mismatch the gate fails, but the golden file is rewritten with the
// new value so the change is visible in the git diff (and accepted by committing
// it) - the test still fails, so the change can't slip in unnoticed.
ReferenceCheck checkReference(const fs::path& path, double actual, double tolerance, bool update) {
    constexpr auto precision = std::numeric_limits<double>::digits10 + 1;
    std::error_code ec;
    if (update || !fs::exists(path, ec)) {
        fs::create_directories(path.parent_path(), ec);
        std::ofstream file{path};
        file << std::setprecision(precision) << actual;
        return {true, true, actual};
    }
    double reference = 0.;
    {
        std::ifstream file{path};
        file >> reference;
    }
    const auto passed = std::abs(actual - reference) <= tolerance;
    if (!passed) {
        std::ofstream file{path};
        file << std::setprecision(precision) << actual;
    }
    return {passed, false, reference};
}

#ifdef NDEBUG
// CPU time (seconds) consumed by the calling thread so far. Timing process() this
// way rather than with a wall clock keeps the benchmark's multithreading from
// inflating the measurement through core contention. POSIX (Linux/macOS); falls
// back to a wall clock elsewhere.
double threadCpuSeconds() {
#if defined(__linux__) || defined(__APPLE__)
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#else
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

// Trimmed stdout of `git <args>`, or "unknown" if git is unavailable / fails.
std::string gitOutput(const std::string& args) {
    std::string out;
    if (FILE* pipe = popen(("git " + args + " 2>/dev/null").c_str(), "r")) {
        char buffer[512];
        while (std::fgets(buffer, sizeof buffer, pipe) != nullptr) {
            out += buffer;
        }
        pclose(pipe);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out.empty() ? "unknown" : out;
}

// Host description (CPU model + logical core count) so RT% rows can be compared
// like-for-like - the measurement is hardware-dependent. Linux /proc/cpuinfo; other
// platforms fall back to just the core count.
std::string machineDescription() {
    std::string cpu = "unknown CPU";
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            const auto start = colon == std::string::npos
                                   ? std::string::npos
                                   : line.find_first_not_of(" \t", colon + 1);
            if (start != std::string::npos) {
                cpu = line.substr(start);
            }
            break;
        }
    }
    return cpu + " (" + std::to_string(std::thread::hardware_concurrency()) + " threads)";
}
#endif  // NDEBUG
}  // namespace

// Benchmarks one algorithm per run, selected with `algorithm=<id>` (defaults to
// the in-house algorithm). Each algorithm declares its own pass/fail gates (see
// getBenchmarkAlgorithms); the reference values are golden files seeded on first
// run and compared within tolerance afterwards.
TEST(PitchDetectorImpl, benchmarking) {
    std::cout << "\n";

    const auto logFilePath = testUtils::getOutDir() / "benchmarking.log";
    std::ofstream logFile(logFilePath);
    testUtils::TeeStream tee(std::cout, logFile);

    const auto argIndexOfProcessToLog = getArgument<int>("indexOfProcessToLog");
    const auto argTestCaseId = getArgument<std::string>("testCaseId");
    const auto argTestWithMedianFilter = getArgument<bool>("testWithMedianFilter");
    // Calibration aid: bypass the probNotOctaviated gate so every frame emits an
    // estimate, yielding the full presence-score/error distribution that
    // eval/fitAndShowErrorProbabilityModels.py needs to re-fit the gate. Pair with
    // testWithMedianFilter=false to get raw per-frame (score, error) pairs.
    const auto argDisableOctaviationGate =
        getArgument<bool>("disableOctaviationGate").value_or(false);
    // Gate-tuning knobs for sweeping #4 without rebuilds (see BenchmarkAlgorithmContext).
    const auto argPresenceThreshold = getArgument<float>("presenceThreshold");
    const auto argHarmonicityFloor = getArgument<float>("harmonicityFloor");
    const auto argThresholdWithEstimateConstraint =
        getArgument<float>("thresholdWithEstimateConstraint");
    const auto argOnsetK = getArgument<float>("onsetK");
    const auto argOnsetAbsFloor = getArgument<float>("onsetAbsFloor");
    const auto argMedianFilterDuration = getArgument<float>("medianFilterDuration");
    const auto argHoldDuration = getArgument<float>("holdDuration");
    // Below-range ("too low") detection knobs (see LowBandConfig).
    const auto argLowBandHarmonicSupportFloor = getArgument<float>("lowBandHarmonicSupportFloor");
    const auto argLowBandMaxHarmonic = getArgument<int>("lowBandMaxHarmonic");
    const auto argLowBandMinFreqRatio = getArgument<float>("lowBandMinFreqRatio");
    const auto argLowBandMinFrames = getArgument<int>("lowBandMinFrames");
    const auto argOutRangePresenceThreshold = getArgument<float>("outRangePresenceThreshold");
    const auto argAlgorithm = getArgument<std::string>("algorithm");
    const auto argDecimationFactor = getArgument<int>("decimationFactor");
    const auto argMinFreqSemitoneOffset = getArgument<int>("minFreqSemitoneOffset");
    const auto updateReferences = getArgument<bool>("updateBenchmarkReferences").value_or(false);

    const auto algorithmId = argAlgorithm.value_or(kDefaultAlgorithmId);
    const auto& algorithms = getBenchmarkAlgorithms();
    ASSERT_TRUE(algorithms.count(algorithmId) > 0) << "Unknown algorithm: " << algorithmId;

    std::cout << "Selected algorithm: " << algorithmId << "\n";
    const auto& algorithm = algorithms.at(algorithmId);
    const auto& createDetector = algorithm.create;

    // Output files of the default algorithm keep their historical names
    // (eval/show*.py import them by module name); other algorithms get a
    // suffix, so that runs of different algorithms can be compared.
    const auto fileSuffix = algorithmId == kDefaultAlgorithmId ? std::string{} : "_" + algorithmId;

    std::optional<std::ofstream> csvFile;

    if (!argTestCaseId.has_value()) {
        const auto csvFilePath = testUtils::getOutDir() / ("benchmarking" + fileSuffix + ".csv");
        csvFile.emplace(csvFilePath);
        *csvFile << "algorithm,AVG,RMS,FPR,FNR,bucketErrorRate,mix,id\n";
    }

    // Build all test cases upfront
    const std::vector<TestCase> testCases = prepareTestCases(argTestCaseId);

    const auto numEvaluations = testCases.size();

    // Pre-allocate results vector for thread-safe indexed access
    std::vector<TestResult> results(testCases.size());
    std::atomic<int> completedCount{0};
    std::mutex progressMutex;

    std::cout << std::endl << "Evaluating samples..." << std::endl;

    // Worker function that processes a range of test cases
    auto processTestCases = [&](size_t startIdx, size_t endIdx) {
        for (size_t idx = startIdx; idx < endIdx; ++idx) {
            const auto& testCase = testCases[idx];
            const auto& sample = testCase.sample;
            const auto& noisy = testCase.noisy;
            const auto blockSize = testCase.blockSize;

            // Each config member starts at its production default (PitchDetectorTypes.h); below
            // we apply the benchmark's one deliberate deviation (hold off, see
            // benchmarkHoldDuration) and any CLI overrides, leaving untouched fields at default.
            BenchmarkAlgorithmContext context;
            context.sampleRate = noisy.sampleRate;
            context.channelFormat = noisy.channelFormat;
            context.blockSize = blockSize;
            context.tuning = kTestTuning;
            context.decimationFactor = argDecimationFactor.value_or(defaultDecimationFactor);
            context.indexOfProcessToLog = argIndexOfProcessToLog;
            context.withMedianFilter =
                !argTestWithMedianFilter.has_value() || *argTestWithMedianFilter;
            context.gate.apply = !argDisableOctaviationGate;
            // Hold off by default so the golden refs reflect the detector's intrinsic behaviour,
            // not the hold's note-persistence. Override with holdDuration=<s>.
            context.hold.holdDuration = benchmarkHoldDuration;
            // CLI overrides to sweep the operating point without rebuilds (see
            // BenchmarkAlgorithmContext).
            if (argPresenceThreshold)
                context.gate.presenceThreshold = *argPresenceThreshold;
            if (argHarmonicityFloor)
                context.gate.harmonicityFloor = *argHarmonicityFloor;
            if (argThresholdWithEstimateConstraint)
                context.gate.presenceThresholdWithConstraint = *argThresholdWithEstimateConstraint;
            if (argOnsetK)
                context.onset.k = *argOnsetK;
            if (argOnsetAbsFloor)
                context.onset.absFloor = *argOnsetAbsFloor;
            if (argMedianFilterDuration)
                context.medianFilter.filterDuration = *argMedianFilterDuration;
            if (argHoldDuration)
                context.hold.holdDuration = *argHoldDuration;
            if (argMinFreqSemitoneOffset)
                context.minFreqSemitoneOffset = *argMinFreqSemitoneOffset;
            if (argLowBandHarmonicSupportFloor)
                context.lowBand.harmonicSupportFloor = *argLowBandHarmonicSupportFloor;
            if (argLowBandMaxHarmonic)
                context.lowBand.maxHarmonic = *argLowBandMaxHarmonic;
            if (argLowBandMinFreqRatio)
                context.lowBand.analysisMinFreqRatio = *argLowBandMinFreqRatio;
            if (argLowBandMinFrames)
                context.lowBand.minConsecutiveFrames = *argLowBandMinFrames;
            if (argOutRangePresenceThreshold)
                context.outRange.presenceThreshold = *argOutRangePresenceThreshold;
            const auto pitchDetector = createDetector(context);
            const std::pair<float, float> pitchSearchRange = pitchDetector->pitchSearchRange();

            auto negativeCount = 0;
            auto falseNegativeWeight = 0.;
            auto positiveWeight = 0.;
            auto falsePositiveCount = 0;
            auto bucketErrorWeight = 0.;
            auto bucketTotalWeight = 0.;
            const auto numChannels = noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
            const auto numFrames = noisy.interleaved.size() / numChannels;
            const auto* noisyData = noisy.interleaved.data();

            std::vector<testUtils::ProcessEstimate> testFileEstimates;
            std::unique_ptr<std::vector<float>> debugOutputSignal;
            if (argTestCaseId) {
                debugOutputSignal = std::make_unique<std::vector<float>>();
            }

            std::vector<bool> onsets;
            std::vector<std::optional<PitchBucket>> buckets;          // returned bucket, per block
            std::vector<std::optional<PitchBucket>> expectedBuckets;  // expected bucket, per block
            std::vector<float> xcorrEstimates;  // pre-gate period estimate (Hz)
            std::vector<float> probsNotOctaviated;
            std::vector<float> lowBandEstimates;  // below-range hypothesis (Hz), its low-band
            std::vector<float> lowBandScores;     // periodicity contrast and its harmonic
            std::vector<float> lowBandSupports;   // support in the spectrum
            std::vector<float> lowBandPresences;  // and how periodic the band itself is
            std::vector<float> rawBuckets;        // the verdict before the median filter
            auto caseProcessCpuSeconds = 0.;      // Release only; 0 otherwise
            auto caseAudioSeconds = 0.;

            const auto expectedNoteBucket = getBucket(sample.truth.frequency, pitchSearchRange);

            for (auto i = 0u; i + blockSize < numFrames; i += blockSize) {
                DebugOutput debugOutput;
#ifdef NDEBUG
                const auto cpuT0 = threadCpuSeconds();
#endif
                const PitchDetectionResult finalEstimate = pitchDetector->process(
                    noisyData + i * numChannels, &debugOutput, debugOutputSignal.get());
#ifdef NDEBUG
                caseProcessCpuSeconds += threadCpuSeconds() - cpuT0;
                caseAudioSeconds += static_cast<double>(blockSize) / noisy.sampleRate;
#endif
                xcorrEstimates.push_back(debugOutput["xcorrEstimate"]);
                probsNotOctaviated.push_back(debugOutput["probNotOctaviated"]);
                lowBandEstimates.push_back(debugOutput["lowBandHz"]);
                lowBandScores.push_back(debugOutput["lowBandScore"]);
                lowBandSupports.push_back(debugOutput["lowBandSupport"]);
                lowBandPresences.push_back(debugOutput["lowBandPresence"]);
                rawBuckets.push_back(debugOutput.count("rawBucket") ? debugOutput["rawBucket"]
                                                                    : -1.f);
                const auto currentTime =
                    static_cast<double>(i + blockSize - pitchDetector->delaySamples()) /
                    noisy.sampleRate;
                const auto truth = (currentTime >= sample.truth.startTime) &&
                                   (currentTime <= sample.truth.endTime);
                auto weight = 0.f;
                if (truth) {
                    // A plucked note decays over its labelled duration, so its SNR
                    // is highest at the onset and lowest by the end. Weight each
                    // positive window linearly from w(startTime) = 1 down to
                    // w(endTime) = 0, so missing a note while it is loud counts for
                    // much more than missing it once it has faded into the noise.
                    weight = (currentTime - sample.truth.endTime) /
                             (sample.truth.startTime - sample.truth.endTime);
                    positiveWeight += weight;
                    if (!finalEstimate.bucket.has_value())
                        falseNegativeWeight += weight;
                } else {
                    ++negativeCount;
                    if (finalEstimate.bucket.has_value())
                        ++falsePositiveCount;
                }
                const auto expectedBucket =
                    truth ? std::make_optional(expectedNoteBucket) : std::nullopt;

                // Only add to bucket error when both actual and expected are pitched. The cases
                // where the one is pitched and the other is not are already covered by FNR and FPR.
                if (finalEstimate.bucket.has_value() && expectedBucket.has_value()) {
                    bucketTotalWeight += weight;
                    if (finalEstimate.bucket != expectedBucket)
                        bucketErrorWeight += weight;
                }

                const auto errorCents =
                    finalEstimate.bucket == PitchBucket::inRange
                        ? 1200.f * std::log2(finalEstimate.pitch / sample.truth.frequency)
                        : 0.f;
                testFileEstimates.emplace_back(weight, debugOutput["presenceScore"],
                                               finalEstimate.pitch, finalEstimate.bucket,
                                               errorCents, debugOutput["harmonicity"]);
                onsets.push_back(debugOutput["isOnset"] == 1.f);
                buckets.push_back(finalEstimate.bucket);
                expectedBuckets.push_back(expectedBucket);
            }

            const auto FPR = 1. * falsePositiveCount / negativeCount;
            const auto FNR = falseNegativeWeight / positiveWeight;

            std::vector<float> frequencyEstimates(testFileEstimates.size());
            std::transform(testFileEstimates.begin(), testFileEstimates.end(),
                           frequencyEstimates.begin(),
                           [](const testUtils::ProcessEstimate& e) { return e.f; });

            // Out-of-range truth: no precise estimate is expected, so the case must not
            // sample the pitch-error stats (FNR/FPR and the bucket error rate still
            // apply to it).
            const std::optional<testUtils::Cents> cents =
                expectedNoteBucket == PitchBucket::inRange
                    ? testUtils::getError(sample.truth.frequency, frequencyEstimates, buckets)
                    : std::nullopt;

            const fs::path cleanFile = testUtils::getFileShortName(sample.file);
            const auto filename = cleanFile.string() + "_with_" +
                                  testCase.noise.file.stem().string() + "_at_" +
                                  testCase.noise.rmsDb + "dB";
            const auto outWavName = (testUtils::getOutDir() / "wav" / filename).string();

            const auto displayCents = cents.value_or(testUtils::Cents{0.f, 0.f});
            const auto evalDir = testUtils::getEvalDir();
            std::stringstream csvLine;
            const auto bucketErrorRate =
                bucketTotalWeight > 0. ? bucketErrorWeight / bucketTotalWeight : 0.;
            csvLine << algorithmId << "," << displayCents.avg << "," << displayCents.rms << ","
                    << FPR << "," << FNR << "," << bucketErrorRate << ","
                    << fs::relative(outWavName, evalDir) << "," << testCase.id << "\n";

            if (argTestCaseId.has_value()) {
                std::cout << csvLine.str();

                // Per-frame diagnostic dump: pre-gate period vs. presence/octaviation
                // gate vs. final estimate. Lets us see whether averaging moved the
                // winning ACF peak or merely raised the gate score on an existing one.
                std::ofstream frameDump(testUtils::getOutDir() /
                                        ("frameDump" + fileSuffix + ".csv"));
                frameDump
                    << "frame,isOnset,presenceScore,probNotOctaviated,xcorrEstimateHz,"
                       "lowBandHz,lowBandScore,lowBandSupport,lowBandPresence,rawBucket,finalHz,"
                       "truthHz,errorCents,"
                       "bucket,"
                       "expectedBucket\n";
                for (size_t f = 0; f < testFileEstimates.size(); ++f) {
                    const auto& e = testFileEstimates[f];
                    frameDump << f << "," << (onsets[f] ? 1 : 0) << "," << e.s << ","
                              << probsNotOctaviated[f] << "," << xcorrEstimates[f] << ","
                              << lowBandEstimates[f] << "," << lowBandScores[f] << ","
                              << lowBandSupports[f] << "," << lowBandPresences[f] << ","
                              << rawBuckets[f] << "," << e.f << "," << sample.truth.frequency << ","
                              << e.e << ","
                              << static_cast<int>(buckets[f].value_or(static_cast<PitchBucket>(-1)))
                              << ","
                              << static_cast<int>(
                                     expectedBuckets[f].value_or(static_cast<PitchBucket>(-1)))
                              << "\n";
                }

                if (argIndexOfProcessToLog.has_value()) {
                    // The logger closed PitchDetectorLog.py at the end of the frame it recorded;
                    // append what only the harness knows, so a plot of that log can place the
                    // frame in the case and judge the estimate against the truth (see
                    // eval/showLowBandAnalysis.py).
                    std::ofstream log(testUtils::getOutDir() / "PitchDetectorLog.py",
                                      std::ios::app);
                    log << "loggedFrame = " << *argIndexOfProcessToLog << "\n";
                    log << "truthHz = " << sample.truth.frequency << "\n";
                    log << "searchRangeHz = [" << pitchSearchRange.first << ", "
                        << pitchSearchRange.second << "]\n";
                }

                std::ofstream frequencyEstimatesFile(testUtils::getOutDir() /
                                                     ("frequencyEstimates" + fileSuffix + ".py"));
                testUtils::PrintPythonVector(frequencyEstimatesFile, frequencyEstimates,
                                             "frequencyEstimates");
                testUtils::PrintPythonVector(frequencyEstimatesFile, onsets, "onsets");
                frequencyEstimatesFile
                    << "secondsPerBlock = " << static_cast<float>(blockSize) / noisy.sampleRate
                    << "\n";

                testUtils::toWavFile(outWavName + "_preprocessed" + fileSuffix + ".wav",
                                     testUtils::Audio{std::move(*debugOutputSignal),
                                                      noisy.sampleRate, noisy.channelFormat},
                                     &tee, "Preprocessed signal");
                testUtils::toWavFile(outWavName + ".wav", noisy, &tee, "Noisy input");
                std::vector<float> presenceSores(testFileEstimates.size());
                std::transform(
                    testFileEstimates.begin(), testFileEstimates.end(), presenceSores.begin(),
                    [](const testUtils::ProcessEstimate& estimate) { return estimate.s; });
                testUtils::toWavFile(
                    testUtils::getOutDir() / ("presenceScores" + fileSuffix + ".wav"),
                    testUtils::Audio{std::move(presenceSores), noisy.sampleRate / blockSize,
                                     ChannelFormat::Mono},
                    &tee, "Presence");
            }

            results[idx] = TestResult{testCase.id,
                                      std::move(testFileEstimates),
                                      cents,
                                      positiveWeight,
                                      negativeCount,
                                      falsePositiveCount,
                                      falseNegativeWeight,
                                      bucketErrorWeight,
                                      bucketTotalWeight,
                                      FPR,
                                      FNR,
                                      csvLine.str(),
                                      sample.file,
                                      testCase.noise.file,
                                      testCase.noise.rmsDb,
                                      caseProcessCpuSeconds,
                                      caseAudioSeconds};

            // Progress reporting (thread-safe)
            const auto completed = ++completedCount;
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                std::cout << "\r" << completed << "/" << numEvaluations << std::flush;
            }
        }
    };

    // Manual threading: split work across available cores
    const auto numThreads = std::thread::hardware_concurrency();
    const auto chunkSize = (testCases.size() + numThreads - 1) / numThreads;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < numThreads; ++t) {
        const auto startIdx = t * chunkSize;
        const auto endIdx = std::min(startIdx + chunkSize, testCases.size());
        if (startIdx < testCases.size()) {
            threads.emplace_back(processTestCases, startIdx, endIdx);
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "\n";

    if (csvFile) {
        for (const auto& result : results) {
            *csvFile << result.csvLine;
        }
    }

    {
        // For histogram
        std::ofstream errorsFile(testUtils::getOutDir() / ("errors" + fileSuffix + ".py"));
        std::vector<float> errors;
        std::vector<float> scores;
        std::vector<float> harmonicities;
        for (const auto& result : results) {
            if (result.cents.has_value()) {
                for (const auto& estimate : result.estimates) {
                    if (estimate.f > 0.f) {
                        errors.push_back(estimate.e);
                        scores.push_back(estimate.s);
                        harmonicities.push_back(estimate.h);
                    }
                }
            }
        }
        testUtils::PrintPythonVector(errorsFile, errors, "errors");
        testUtils::PrintPythonVector(errorsFile, scores, "scores");
        testUtils::PrintPythonVector(errorsFile, harmonicities, "harmonicities");
    }

    if (argTestCaseId.has_value()) {
        return;
    }

    auto totalPositiveWeight = 0.;
    auto totalNegativeCount = 0;
    auto totalFalsePositiveCount = 0;
    auto totalFalseNegativeWeight = 0.;
    auto totalBucketErrorWeight = 0.;
    auto totalBucketWeight = 0.;

    auto avgAvg = 0.;
    auto rmsAvg = 0.;
    auto count = 0;
    auto worstRms = 0.;
    auto worstRmsIndex = 0;
    std::vector<double> perCaseRms;
    for (auto i = 0u; i < results.size(); ++i) {
        const auto& result = results[i];
        totalPositiveWeight += result.positiveWeight;
        totalNegativeCount += result.negativeCount;
        totalFalsePositiveCount += result.falsePositiveCount;
        totalFalseNegativeWeight += result.falseNegativeWeight;
        totalBucketErrorWeight += result.bucketErrorWeight;
        totalBucketWeight += result.bucketTotalWeight;
        if (result.cents.has_value()) {
            ++count;
            avgAvg += result.cents->avg;
            rmsAvg += result.cents->rms;
            perCaseRms.push_back(result.cents->rms);
            if (result.cents->rms > worstRms) {
                worstRms = result.cents->rms;
                worstRmsIndex = i;
            }
        }
    }
    avgAvg /= count;
    rmsAvg /= count;
    // Median and 99th-percentile per-case RMS. Both are robust to the *magnitude* of
    // the catastrophic outliers (an octave vs. two-octave error reads the same in the
    // tuner UI), unlike the mean: the median tracks typical accuracy, the 99th
    // percentile the tail (how often catastrophic errors occur).
    auto medianRms = 0.;
    auto p99Rms = 0.;
    if (!perCaseRms.empty()) {
        std::sort(perCaseRms.begin(), perCaseRms.end());
        const auto mid = perCaseRms.size() / 2;
        medianRms = perCaseRms.size() % 2 == 0 ? 0.5 * (perCaseRms[mid - 1] + perCaseRms[mid])
                                               : perCaseRms[mid];
        // Linear-interpolated percentile (matches numpy's default).
        const auto rank = 0.99 * (perCaseRms.size() - 1);
        const auto lo = static_cast<size_t>(std::floor(rank));
        const auto hi = static_cast<size_t>(std::ceil(rank));
        p99Rms = perCaseRms[lo] + (rank - lo) * (perCaseRms[hi] - perCaseRms[lo]);
    }

    const auto globalFalsePositiveRate = 1. * totalFalsePositiveCount / totalNegativeCount;
    const auto globalFalseNegativeRate = totalFalseNegativeWeight / totalPositiveWeight;
    const auto globalBucketErrorRate =
        totalBucketWeight > 0. ? totalBucketErrorWeight / totalBucketWeight : 0.;

    tee << "[" << algorithmId << "] Error across all tests:\n\tAVG: " << avgAvg
        << "\n\tRMS: " << rmsAvg << "\n\tmedian RMS: " << medianRms
        << "\n\t99th-pct RMS: " << p99Rms << "\n\tFPR: " << globalFalsePositiveRate
        << "\n\tFNR: " << globalFalseNegativeRate
        << "\n\tbucket error rate: " << globalBucketErrorRate << "\n\tworst RMS error: " << worstRms
        << " at index " << worstRmsIndex << " (" << results[worstRmsIndex].id << ")\n";

    std::vector<testUtils::ProcessEstimate> estimatesForRoc;
    for (const auto& result : results) {
        estimatesForRoc.insert(estimatesForRoc.end(), result.estimates.begin(),
                               result.estimates.end());
    }

    constexpr auto allowedFalsePositiveRate = 0.01;  // 1%
    const testUtils::RocInfo rocInfo = testUtils::GetRocInfo<testUtils::ProcessEstimate>(
        estimatesForRoc, allowedFalsePositiveRate);

    {
        std::ofstream rocFile(testUtils::getOutDir() / ("roc_curve" + fileSuffix + ".py"));
        rocFile << "AUC = " << rocInfo.areaUnderCurve << "\n";
        rocFile << "threshold = " << rocInfo.threshold << "\n";
        rocFile << "allowedFalsePositiveRate = " << allowedFalsePositiveRate << "\n";
        testUtils::PrintPythonVector(rocFile, rocInfo.falsePositiveRates, "falsePositiveRates");
        testUtils::PrintPythonVector(rocFile, rocInfo.truePositiveRates, "truePositiveRates");
    }

    // Pass/fail gating. Each algorithm declares which metrics it is gated on (see
    // getBenchmarkAlgorithms); the reference values live in golden files under
    // eval/BenchmarkingOutput, seeded on the first run (or with
    // updateBenchmarkReferences=true) and compared within tolerance afterwards.
    // The in-house algorithm's golden files are committed, so it is gated from the
    // start; a brand-new third-party algorithm seeds its references on first run.
    const BenchmarkMetrics metrics{avgAvg,
                                   rmsAvg,
                                   medianRms,
                                   p99Rms,
                                   globalFalsePositiveRate,
                                   globalFalseNegativeRate,
                                   rocInfo.areaUnderCurve,
                                   globalBucketErrorRate};
    const auto referenceDir = testUtils::getEvalDir() / "BenchmarkingOutput";
    for (const auto& gate : algorithm.gates) {
        const auto refPath = referenceDir / (gate.fileStem + fileSuffix + ".txt");
        const auto actual = gate.value(metrics);
        const auto check = checkReference(refPath, actual, gate.tolerance, updateReferences);
        if (check.seeded) {
            tee << "[" << algorithmId << "] seeded reference " << gate.displayName << " = "
                << actual << " (" << refPath.filename().string() << ")\n";
            continue;
        }
        // A change for the better is probably good, but worth keeping an eye on;
        // a change for the worse is either justified or a regression.
        EXPECT_TRUE(check.passed)
            << "[" << algorithmId << "] " << gate.displayName
            << " has changed! reference: " << check.reference << ", now: " << actual
            << " (tolerance " << gate.tolerance
            << "; rerun with updateBenchmarkReferences=true to accept the new value)";
    }

#ifdef NDEBUG
    // process() CPU cost as a fraction of real time (CPU-seconds per second of audio,
    // in percent). Per-algorithm, machine/load dependent and so NOT gated; appended to
    // a checked-in log keyed by branch+commit so we can compare costs across commits
    // (e.g. how much ACF averaging adds). Release + full-benchmark only.
    {
        auto totalProcessCpuSeconds = 0.;
        auto totalAudioSeconds = 0.;
        for (const auto& result : results) {
            totalProcessCpuSeconds += result.processCpuSeconds;
            totalAudioSeconds += result.audioSeconds;
        }
        const auto realtimePercent =
            totalAudioSeconds > 0. ? 100. * totalProcessCpuSeconds / totalAudioSeconds : 0.;
        tee << "[" << algorithmId << "] process() real-time cost: " << realtimePercent << "%\n";

        std::time_t now = std::time(nullptr);
        char dateBuf[16] = "0000-00-00";
        std::strftime(dateBuf, sizeof dateBuf, "%Y-%m-%d", std::localtime(&now));

        const auto logPath = testUtils::getEvalDir() / "cpu-realtime-log.md";
        const bool existed = fs::exists(logPath);
        std::ofstream cpuLog(logPath, std::ios::app);
        if (!existed) {
            cpuLog << "# process() real-time CPU cost\n\n"
                   << "Release, full-benchmark only. RT% = CPU time in `process()` / audio "
                      "duration x 100,\nmeasured with a per-thread CPU clock (contention-free). "
                      "Machine- and load-dependent\nand not reproducible, so it is recorded here "
                      "for comparison rather than gated.\n\n"
                   << "| date | branch | commit | message | algorithm | machine | RT% |\n"
                   << "|------|--------|--------|---------|-----------|---------|-----|\n";
        }
        cpuLog << "| " << dateBuf << " | " << gitOutput("rev-parse --abbrev-ref HEAD") << " | "
               << gitOutput("rev-parse --short HEAD") << " | " << gitOutput("log -1 --format=%s")
               << " | " << algorithmId << " | " << machineDescription() << " | " << std::fixed
               << std::setprecision(2) << realtimePercent << " |\n";
    }
#endif  // NDEBUG
}
}  // namespace saint
