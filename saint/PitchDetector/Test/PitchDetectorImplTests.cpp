#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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
    std::optional<testUtils::Cents> cents;
    double positiveWeight = 0.;
    int negativeCount = 0;
    int falsePositiveCount = 0;
    double falseNegativeWeight = 0.;
    double FPR = 0.;
    double FNR = 0.;
    std::string csvLine;
    fs::path testFile;
    fs::path noiseFile;
    std::string noiseRmsDb;
};

struct ReferenceCheck {
    bool passed = false;
    bool seeded = false;  // golden file was just created (or refreshed)
    double reference = 0.;
};

// Compares an algorithm metric against its golden reference file. The file is
// seeded with the current value when it does not yet exist (or when `update` is
// set), and otherwise treated as the authority - a mismatch fails the gate
// without silently rewriting the file.
ReferenceCheck checkReference(const fs::path& path, double actual, double tolerance, bool update) {
    std::error_code ec;
    if (update || !fs::exists(path, ec)) {
        fs::create_directories(path.parent_path(), ec);
        std::ofstream file{path};
        file << std::setprecision(std::numeric_limits<double>::digits10 + 1) << actual;
        return {true, true, actual};
    }
    std::ifstream file{path};
    double reference = 0.;
    file >> reference;
    return {std::abs(actual - reference) <= tolerance, false, reference};
}
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
    const auto argAlgorithm = getArgument<std::string>("algorithm");
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
        *csvFile << "algorithm,AVG,RMS,FPR,FNR,mix,id\n";
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

            const BenchmarkAlgorithmContext context{
                noisy.sampleRate,
                noisy.channelFormat,
                blockSize,
                kTestTuning,
                argIndexOfProcessToLog,
                !argTestWithMedianFilter.has_value() || *argTestWithMedianFilter};
            const auto pitchDetector = createDetector(context);

            auto negativeCount = 0;
            auto falseNegativeWeight = 0.;
            auto positiveWeight = 0.;
            auto falsePositiveCount = 0;
            const auto numChannels = noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
            const auto numFrames = noisy.interleaved.size() / numChannels;
            const auto* noisyData = noisy.interleaved.data();

            std::vector<testUtils::ProcessEstimate> testFileEstimates;
            std::unique_ptr<std::vector<float>> debugOutputSignal;
            if (argTestCaseId) {
                debugOutputSignal = std::make_unique<std::vector<float>>();
            }

            std::vector<bool> onsets;

            for (auto i = 0u; i + blockSize < numFrames; i += blockSize) {
                DebugOutput debugOutput;
                const auto finalEstimate = pitchDetector->process(
                    noisyData + i * numChannels, &debugOutput, debugOutputSignal.get());
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
                    if (finalEstimate == 0.f)
                        falseNegativeWeight += weight;
                } else {
                    ++negativeCount;
                    if (finalEstimate != 0.f)
                        ++falsePositiveCount;
                }
                const auto errorCents =
                    finalEstimate > 0.f ? 1200.f * std::log2(finalEstimate / sample.truth.frequency)
                                        : 0.f;
                testFileEstimates.emplace_back(weight, debugOutput["presenceScore"], finalEstimate,
                                               errorCents);
                onsets.push_back(debugOutput["isOnset"] == 1.f);
            }

            const auto FPR = 1. * falsePositiveCount / negativeCount;
            const auto FNR = falseNegativeWeight / positiveWeight;

            std::vector<float> frequencyEstimates(testFileEstimates.size());
            std::transform(testFileEstimates.begin(), testFileEstimates.end(),
                           frequencyEstimates.begin(),
                           [](const testUtils::ProcessEstimate& e) { return e.f; });

            const std::optional<testUtils::Cents> cents =
                testUtils::getError(sample.truth.frequency, frequencyEstimates);

            const fs::path cleanFile = testUtils::getFileShortName(sample.file);
            const auto filename = cleanFile.string() + "_with_" +
                                  testCase.noise.file.stem().string() + "_at_" +
                                  testCase.noise.rmsDb + "dB";
            const auto outWavName = (testUtils::getOutDir() / "wav" / filename).string();

            const auto displayCents = cents.value_or(testUtils::Cents{0.f, 0.f});
            const auto evalDir = testUtils::getEvalDir();
            std::stringstream csvLine;
            csvLine << algorithmId << "," << displayCents.avg << "," << displayCents.rms << ","
                    << FPR << "," << FNR << "," << fs::relative(outWavName, evalDir) << ","
                    << testCase.id << "\n";

            if (argTestCaseId.has_value()) {
                std::cout << csvLine.str();

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
                                      FPR,
                                      FNR,
                                      csvLine.str(),
                                      sample.file,
                                      testCase.noise.file,
                                      testCase.noise.rmsDb};

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
        for (const auto& result : results) {
            if (result.cents.has_value()) {
                for (const auto& estimate : result.estimates) {
                    if (estimate.f > 0.f) {
                        errors.push_back(estimate.e);
                        scores.push_back(estimate.s);
                    }
                }
            }
        }
        testUtils::PrintPythonVector(errorsFile, errors, "errors");
        testUtils::PrintPythonVector(errorsFile, scores, "scores");
    }

    if (argTestCaseId.has_value()) {
        return;
    }

    auto totalPositiveWeight = 0.;
    auto totalNegativeCount = 0;
    auto totalFalsePositiveCount = 0;
    auto totalFalseNegativeWeight = 0.;

    auto avgAvg = 0.;
    auto rmsAvg = 0.;
    auto count = 0;
    auto worstRms = 0.;
    auto worstRmsIndex = 0;
    for (auto i = 0u; i < results.size(); ++i) {
        const auto& result = results[i];
        totalPositiveWeight += result.positiveWeight;
        totalNegativeCount += result.negativeCount;
        totalFalsePositiveCount += result.falsePositiveCount;
        totalFalseNegativeWeight += result.falseNegativeWeight;
        if (result.cents.has_value()) {
            ++count;
            avgAvg += result.cents->avg;
            rmsAvg += result.cents->rms;
            if (result.cents->rms > worstRms) {
                worstRms = result.cents->rms;
                worstRmsIndex = i;
            }
        }
    }
    avgAvg /= count;
    rmsAvg /= count;

    const auto globalFalsePositiveRate = 1. * totalFalsePositiveCount / totalNegativeCount;
    const auto globalFalseNegativeRate = totalFalseNegativeWeight / totalPositiveWeight;

    tee << "[" << algorithmId << "] Error across all tests:\n\tAVG: " << avgAvg
        << "\n\tRMS: " << rmsAvg << "\n\tFPR: " << globalFalsePositiveRate
        << "\n\tFNR: " << globalFalseNegativeRate << "\n\tworst RMS error: " << worstRms
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
    const BenchmarkMetrics metrics{avgAvg, rmsAvg, globalFalsePositiveRate, globalFalseNegativeRate,
                                   rocInfo.areaUnderCurve};
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
}
}  // namespace saint
