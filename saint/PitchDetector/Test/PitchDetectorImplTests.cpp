#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "AutocorrPitchDetector.h"
#include "DummyPitchDetectorLogger.h"
#include "OnsetDetector.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorImplTestWrapper.h"
#include "PitchDetectorLogger.h"
#include "PitchDetectorMedianFilter.h"
#include "PitchDetectorUtils.h"
#include "TestCaseUtils.h"
#include "Utils.h"
#include "testUtils.h"

namespace saint {
namespace fs = std::filesystem;
namespace {
struct TestResult {
    std::string id;
    std::vector<testUtils::ProcessEstimate> estimates;
    std::optional<testUtils::Cents> cents;
    double FPR;
    double FNR;
    std::string csvLine;
    fs::path testFile;
    fs::path noiseFile;
    std::string noiseRmsDb;
};
}  // namespace

TEST(PitchDetectorImpl, benchmarking) {
    std::cout << "\n";

    const auto logFilePath = testUtils::getOutDir() / "benchmarking.log";
    std::ofstream logFile(logFilePath);
    testUtils::TeeStream tee(std::cout, logFile);

    const auto argIndexOfProcessToLog = getArgument<int>("indexOfProcessToLog");
    const auto argTestCaseId = getArgument<std::string>("testCaseId");
    const auto argTestWithMedianFilter = getArgument<bool>("testWithMedianFilter");

    std::optional<std::ofstream> csvFile;

    if (!argTestCaseId.has_value()) {
        const auto csvFilePath = testUtils::getOutDir() / "benchmarking.csv";
        csvFile.emplace(csvFilePath);
        *csvFile << "AVG,RMS,FPR,FNR,mix,id\n";
    }

    // Build all test cases upfront
    const std::vector<TestCase> testCases = prepareTestCases(argTestCaseId);

    const auto numEvaluations = testCases.size();

    // Pre-allocate results vector for thread-safe indexed access
    std::vector<TestResult> results(testCases.size());
    std::atomic<int> completedCount{0};
    std::mutex progressMutex;

    auto totalPositiveCount = 0;
    auto totalNegativeCount = 0;
    auto totalFalsePositiveCount = 0;
    auto totalFalseNegativeCount = 0;

    std::cout << std::endl << "Evaluating samples..." << std::endl;

    // Worker function that processes a range of test cases
    auto processTestCases = [&](size_t startIdx, size_t endIdx) {
        for (size_t idx = startIdx; idx < endIdx; ++idx) {
            const auto& testCase = testCases[idx];
            const auto& sample = testCase.sample;
            const auto& noisy = testCase.noisy;
            const auto blockSize = testCase.blockSize;

            std::unique_ptr<PitchDetectorLoggerInterface> logger;
            if (argIndexOfProcessToLog.has_value()) {
                logger = std::make_unique<PitchDetectorLogger>(noisy.sampleRate,
                                                               *argIndexOfProcessToLog);
            } else {
                logger = std::make_unique<DummyPitchDetectorLogger>();
            }

            const auto minFreq = getMinFreq(kTestConfig);
            auto preprocessor =
                std::make_unique<Preprocessor>(noisy.sampleRate, noisy.channelFormat, blockSize);

            FrequencyDomainTransformer transformer(noisy.sampleRate, noisy.channelFormat, blockSize,
                                                   minFreq, *logger);
            AutocorrPitchDetector autocorrPitchDetector(noisy.sampleRate, transformer.fftSize(),
                                                        transformer.window(), minFreq, *logger);
            AutocorrEstimateDisambiguator disambiguator(noisy.sampleRate, transformer.fftSize(),
                                                        kTestConfig, *logger);
            OnsetDetector onsetDetector(noisy.sampleRate, noisy.channelFormat, blockSize, minFreq);

            auto internalAlgorithm = std::make_unique<PitchDetectorImpl>(
                std::move(preprocessor), std::move(transformer), std::move(autocorrPitchDetector),
                std::move(disambiguator), std::move(onsetDetector), std::move(logger));
            std::unique_ptr<PitchDetector> pitchDetector;

            if (!argTestWithMedianFilter.has_value() || *argTestWithMedianFilter) {
                pitchDetector = std::make_unique<PitchDetectorMedianFilter>(
                    noisy.sampleRate, blockSize, std::move(internalAlgorithm));
            } else {
                pitchDetector =
                    std::make_unique<PitchDetectorImplTestWrapper>(std::move(internalAlgorithm));
            }

            auto negativeCount = 0;
            auto falseNegativeCount = 0;
            auto positiveCount = 0;
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
                if (truth) {
                    ++positiveCount;
                    if (finalEstimate == 0.f)
                        ++falseNegativeCount;
                } else {
                    ++negativeCount;
                    if (finalEstimate != 0.f)
                        ++falsePositiveCount;
                }
                const auto errorCents =
                    finalEstimate > 0.f ? 1200.f * std::log2(finalEstimate / sample.truth.frequency)
                                        : 0.f;
                testFileEstimates.emplace_back(truth, debugOutput["presenceScore"], finalEstimate,
                                               errorCents);
                onsets.push_back(debugOutput["isOnset"] == 1.f);
            }

            totalPositiveCount += positiveCount;
            totalNegativeCount += negativeCount;
            totalFalsePositiveCount += falsePositiveCount;
            totalFalseNegativeCount += falseNegativeCount;

            const auto FPR = 1. * falsePositiveCount / negativeCount;
            const auto FNR = 1. * falseNegativeCount / positiveCount;

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
            csvLine << displayCents.avg << "," << displayCents.rms << "," << FPR << "," << FNR
                    << "," << fs::relative(outWavName, evalDir) << "," << testCase.id << "\n";

            if (argTestCaseId.has_value()) {
                std::cout << csvLine.str();

                std::ofstream frequencyEstimatesFile(testUtils::getOutDir() /
                                                     "frequencyEstimates.py");
                testUtils::PrintPythonVector(frequencyEstimatesFile, frequencyEstimates,
                                             "frequencyEstimates");
                testUtils::PrintPythonVector(frequencyEstimatesFile, onsets, "onsets");
                frequencyEstimatesFile
                    << "secondsPerBlock = " << static_cast<float>(blockSize) / noisy.sampleRate
                    << "\n";

                testUtils::toWavFile(outWavName + "_preprocessed.wav",
                                     testUtils::Audio{std::move(*debugOutputSignal),
                                                      noisy.sampleRate, noisy.channelFormat},
                                     &tee, "Preprocessed signal");
                testUtils::toWavFile(outWavName + ".wav", noisy, &tee, "Noisy input");
                std::vector<float> presenceSores(testFileEstimates.size());
                std::transform(
                    testFileEstimates.begin(), testFileEstimates.end(), presenceSores.begin(),
                    [](const testUtils::ProcessEstimate& estimate) { return estimate.s; });
                testUtils::toWavFile(
                    testUtils::getOutDir() / "presenceScores.wav",
                    testUtils::Audio{std::move(presenceSores), noisy.sampleRate / blockSize,
                                     ChannelFormat::Mono},
                    &tee, "Presence");
            }

            results[idx] = TestResult{testCase.id,
                                      std::move(testFileEstimates),
                                      cents,
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

    // Aggregate results (sequential)
    std::vector<testUtils::ProcessEstimate> estimatesForRoc;
    std::vector<std::optional<testUtils::Cents>> allTestFileEstimates;

    for (const auto& result : results) {
        if (csvFile)
            *csvFile << result.csvLine;
        allTestFileEstimates.push_back(result.cents);
        estimatesForRoc.insert(estimatesForRoc.end(), result.estimates.begin(),
                               result.estimates.end());
    }

    {
        // For histogram
        std::ofstream errorsFile(testUtils::getOutDir() / "errors.py");
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

    auto avgAvg = 0.;
    auto rmsAvg = 0.;
    auto count = 0;
    auto worstRms = 0.;
    auto worstRmsIndex = 0;
    for (auto i = 0; i < allTestFileEstimates.size(); ++i) {
        const auto& e = allTestFileEstimates[i];
        if (e.has_value()) {
            ++count;
            avgAvg += e->avg;
            rmsAvg += e->rms;
            if (e->rms > worstRms) {
                worstRms = e->rms;
                worstRmsIndex = i;
            }
        }
    }
    avgAvg /= count;
    rmsAvg /= count;

    const auto globalFalsePositiveRate = 1. * totalFalsePositiveCount / totalNegativeCount;
    const auto globalFalseNegativeRate = 1. * totalFalseNegativeCount / totalPositiveCount;

    tee << "Error across all tests:\n\tAVG: " << avgAvg << "\n\tRMS: " << rmsAvg
        << "\n\tFPR: " << globalFalsePositiveRate << "\n\tFNR: " << globalFalseNegativeRate
        << "\n\tworst RMS error: " << worstRms << " at index " << worstRmsIndex << "\n";

    constexpr auto previousRmsError = 13.96999747265438;
    constexpr auto previousAuc = 0.8815561328790874;
    constexpr auto previousFNR = 0.4210149117900784;

    constexpr auto comparisonTolerance = 0.01;

    const auto fnrIsUnchanged =
        testUtils::valueIsUnchanged(testUtils::getEvalDir() / "BenchmarkingOutput" / "FNR.txt",
                                    previousFNR, globalFalseNegativeRate, comparisonTolerance);
    EXPECT_TRUE(fnrIsUnchanged) << "False Negative Rate has changed! Previous FNR: " << previousFNR
                                << ", new FNR: " << globalFalseNegativeRate;

    const auto rmsErrorIsUnchanged = testUtils::valueIsUnchanged(
        testUtils::getEvalDir() / "BenchmarkingOutput" / "RMS_error.txt", previousRmsError, rmsAvg,
        comparisonTolerance);
    EXPECT_TRUE(rmsErrorIsUnchanged)
        << "RMS error has changed! Previous RMS error: " << previousRmsError
        << ", new RMS error: " << rmsAvg;

    constexpr auto allowedFalsePositiveRate = 0.01;  // 1%
    const testUtils::RocInfo rocInfo = testUtils::GetRocInfo<testUtils::ProcessEstimate>(
        estimatesForRoc, allowedFalsePositiveRate);

    {
        std::ofstream rocFile(testUtils::getOutDir() / "roc_curve.py");
        rocFile << "AUC = " << rocInfo.areaUnderCurve << "\n";
        rocFile << "threshold = " << rocInfo.threshold << "\n";
        rocFile << "allowedFalsePositiveRate = " << allowedFalsePositiveRate << "\n";
        testUtils::PrintPythonVector(rocFile, rocInfo.falsePositiveRates, "falsePositiveRates");
        testUtils::PrintPythonVector(rocFile, rocInfo.truePositiveRates, "truePositiveRates");
    }

    const auto classifierQualityIsUnchanged =
        testUtils::valueIsUnchanged(testUtils::getEvalDir() / "BenchmarkingOutput" / "AUC.txt",
                                    previousAuc, rocInfo.areaUnderCurve, comparisonTolerance);

    // If it changes and it's for the better, then it's probably a good thing, but
    // let's keep an eye on it anyway. If it's for the worse, then either there is
    // a good reason or we have a regression.
    EXPECT_TRUE(classifierQualityIsUnchanged)
        << "Classifier quality has changed! Previous AUC: " << previousAuc
        << ", new AUC: " << rocInfo.areaUnderCurve;
}
}  // namespace saint
