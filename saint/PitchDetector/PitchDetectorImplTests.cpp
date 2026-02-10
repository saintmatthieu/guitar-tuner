#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "AutocorrPitchDetector.h"
#include "DummyPitchDetectorLogger.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorLogger.h"
#include "PitchDetectorMedianFilter.h"
#include "PitchDetectorUtils.h"
#include "Utils.h"
#include "testUtils.h"

namespace saint {

namespace fs = std::filesystem;

namespace {

// Create a test case ID from clean file, noise file, and noise level.
// Format: cleanFilePath|noiseFilePath|dBLevel (using relative paths from eval dir)
std::string computeTestCaseId(const fs::path& cleanFile, const fs::path& noiseFile,
                              const std::string& noiseRmsDb) {
    const auto evalDir = testUtils::getEvalDir();
    const auto relativeClean = fs::relative(cleanFile, evalDir).string();
    const auto relativeNoise = fs::relative(noiseFile, evalDir).string();
    return relativeClean + " | " + relativeNoise + " | " + noiseRmsDb;
}

// Parse a test case ID back into its components
struct ParsedTestCaseId {
    fs::path cleanFile;
    fs::path noiseFile;
    std::string noiseRmsDb;
};

std::optional<ParsedTestCaseId> parseTestCaseId(const std::string& id) {
    const auto evalDir = testUtils::getEvalDir();
    const auto firstSep = id.find(" | ");
    if (firstSep == std::string::npos)
        return std::nullopt;
    const auto secondSep = id.find(" | ", firstSep + 3);
    if (secondSep == std::string::npos)
        return std::nullopt;

    return ParsedTestCaseId{evalDir / id.substr(0, firstSep),
                            evalDir / id.substr(firstSep + 3, secondSep - firstSep - 3),
                            id.substr(secondSep + 3)};
}

struct Noise {
    fs::path file;
    std::string rmsDb;
    std::vector<float> data;
};

struct TestCase {
    std::string id;  // Stable ID based on file paths and noise level
    testUtils::Sample sample;
    Noise noise;
    testUtils::Audio noisy;
    int blockSize;
};

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

std::vector<testUtils::Sample> loadSamples() {
    std::vector<testUtils::Sample> samples;
    const auto& argv = ::testing::internal::GetArgvs();
    for (size_t i = 1; i < argv.size(); ++i) {
        const fs::path testFile(argv[i]);
        if (!fs::exists(testFile) || testFile.extension() != ".wav") {
            continue;
        }
        auto sample = testUtils::getSampleFromFile(testFile);
        if (sample.has_value()) {
            samples.push_back(std::move(*sample));
        }
    }

    if (samples.empty()) {
        const fs::path testFileDir = testUtils::getEvalDir() / "testFiles" / "notes";
        for (const auto& entry : fs::recursive_directory_iterator(testFileDir)) {
            if (entry.path().extension() != ".wav") {
                continue;
            }
            auto sample = testUtils::getSampleFromFile(entry.path());
            if (sample.has_value()) {
                samples.push_back(std::move(*sample));
            }
        }
    }
    // Sort samples by file path to ensure deterministic test case indices
    std::sort(
        samples.begin(), samples.end(),
        [](const testUtils::Sample& a, const testUtils::Sample& b) { return a.file < b.file; });

    return samples;
}

std::vector<Noise> loadNoiseData(int numFrames, const fs::path& silenceFilePath) {
    std::vector<fs::path> noiseFiles;
    for (const auto& entry :
         fs::recursive_directory_iterator(testUtils::getEvalDir() / "testFiles" / "noise")) {
        if (entry.path().extension() == ".wav") {
            noiseFiles.push_back(entry.path());
        }
    }
    // Sort noise files to ensure deterministic test case indices
    std::sort(noiseFiles.begin(), noiseFiles.end());

    std::vector<Noise> noiseData;

    const auto silenceAudio = testUtils::fromWavFile(silenceFilePath, numFrames);
    noiseData.push_back(Noise{silenceFilePath, "-inf", silenceAudio->interleaved});

    const std::vector<std::string> noiseRmsDb{"-40", "-50", "-60"};
    for (const auto& noiseFile : noiseFiles) {
        const auto noiseAudio = testUtils::fromWavFile(noiseFile, numFrames);
        if (!noiseAudio.has_value()) {
            continue;
        }
        for (const auto& rmsDb : noiseRmsDb) {
            const float dB = std::stof(rmsDb);
            auto copy = noiseAudio->interleaved;
            testUtils::scaleToRms(copy, dB);
            noiseData.push_back(Noise{noiseFile, rmsDb, std::move(copy)});
        }
    }

    return noiseData;
}

// Targeting tuning of acoustic guitar:
// - min note accounts for a drop-D tuning and an additional tone to account for
// pitch changes while tuning
// - max note is the high E on the first string, adding a tone for margin
constexpr PitchDetectorConfig config{
    Pitch{PitchClass::Db, 2},
    Pitch{PitchClass::Gb, 4},
};

// Create a single test case from a parsed ID (fast path for debugging)
std::optional<TestCase> createTestCaseFromId(const std::string& testCaseId) {
    const auto parsed = parseTestCaseId(testCaseId);
    if (!parsed.has_value()) {
        std::cerr << "Could not parse test case ID: " << testCaseId << "\n";
        return std::nullopt;
    }

    auto sample = testUtils::getSampleFromFile(parsed->cleanFile);
    if (!sample.has_value()) {
        std::cerr << "Could not get sample from file: " << parsed->cleanFile << "\n";
        return std::nullopt;
    }

    auto clean = testUtils::fromWavFile(parsed->cleanFile);
    if (!clean.has_value()) {
        std::cerr << "Could not read clean file: " << parsed->cleanFile << "\n";
        return std::nullopt;
    }

    const auto blockSize = clean->sampleRate / 100;
    testUtils::scaleToPeak(clean->interleaved, -10.f);

    // Load and scale the noise
    std::vector<float> noiseData;
    if (parsed->noiseRmsDb == "-inf") {
        // Silence
        noiseData.resize(clean->interleaved.size(), 0.f);
    } else {
        // Note: fromWavFile expects number of frames, not samples
        const auto numFrames = clean->numFrames();
        auto noiseAudio = testUtils::fromWavFile(parsed->noiseFile, numFrames);
        if (!noiseAudio.has_value()) {
            std::cerr << "Could not read noise file: " << parsed->noiseFile << "\n";
            return std::nullopt;
        }
        const float dB = std::stof(parsed->noiseRmsDb);
        testUtils::scaleToRms(noiseAudio->interleaved, dB);
        noiseData = std::move(noiseAudio->interleaved);
    }

    auto noisy = *clean;
    testUtils::mixNoise(noisy, noiseData);

    return TestCase{testCaseId, std::move(*sample),
                    Noise{parsed->noiseFile, parsed->noiseRmsDb, std::move(noiseData)},
                    std::move(noisy), blockSize};
}

std::vector<TestCase> prepareTestCases(const std::optional<std::string>& argTestCaseId) {
    // Fast path: if a specific test case ID is provided, parse it and create only that test case
    if (argTestCaseId.has_value()) {
        std::cout << "Creating test case from ID: " << *argTestCaseId << std::endl;
        auto testCase = createTestCaseFromId(*argTestCaseId);
        if (testCase.has_value()) {
            return {std::move(*testCase)};
        }
        std::cerr << "Failed to create test case from ID\n";
        return {};
    }

    // Full path: iterate through all samples and noise combinations
    const auto samples = loadSamples();

    const std::vector<float> silence(44100, 0.f);
    const auto silenceFilePath = testUtils::getOutDir() / "wav" / "silence.wav";
    auto silenceWriter = std::make_shared<testUtils::RealFileWriter>();
    silenceWriter->toWavFile(silenceFilePath, {silence, 44100, ChannelFormat::Mono}, nullptr);

    std::vector<TestCase> testCases;

    std::cout << "Preparing test cases..." << std::endl;
    std::cout << "Number of samples: " << samples.size() << std::endl;

    auto testCaseCount = 0;
    for (const auto& sample : samples) {
        const auto& testFile = sample.file;

        std::cout << "\r" << ++testCaseCount << "/" << samples.size() << std::flush;

        std::optional<testUtils::Audio> clean = testUtils::fromWavFile(testFile);
        if (!clean.has_value()) {
            std::cerr << "Could not read file: " << testFile << "\n";
            continue;
        }

        const auto blockSize = clean->sampleRate / 100;
        testUtils::scaleToPeak(clean->interleaved, -10.f);

        const auto noiseData = loadNoiseData(clean->numFrames(), silenceFilePath);

        for (const auto& noise : noiseData) {
            const auto id = computeTestCaseId(testFile, noise.file, noise.rmsDb);
            auto noisy = *clean;
            testUtils::mixNoise(noisy, noise.data);
            testCases.push_back(TestCase{id, sample, noise, std::move(noisy), blockSize});
        }
    }
    return testCases;
}

template <typename T>
std::optional<T> getArgument(const std::string& prefix) {
    const auto& argv = ::testing::internal::GetArgvs();
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string argStr(argv[i]);
        if (argStr.find(prefix) == 0) {
            if constexpr (std::is_same_v<T, fs::path>) {
                return fs::path(argStr.substr(prefix.size() + 1 /* skip "=" */));
            } else if constexpr (std::is_same_v<T, int>) {
                return std::stoi(argStr.substr(prefix.size() + 1 /* skip "=" */));
            } else if constexpr (std::is_same_v<T, std::string>) {
                return argStr.substr(prefix.size() + 1 /* skip "=" */);
            } else if constexpr (std::is_same_v<T, bool>) {
                const auto valueStr = argStr.substr(prefix.size() + 1 /* skip "=" */);
                return (valueStr == "1" || valueStr == "true" || valueStr == "True");
            }
        }
    }
    return std::nullopt;
}
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

            const auto minFreq = getMinFreq(config);
            FrequencyDomainTransformer transformer(noisy.sampleRate, noisy.channelFormat, blockSize,
                                                   minFreq, *logger);
            AutocorrPitchDetector autocorrPitchDetector(noisy.sampleRate, transformer.fftSize(),
                                                        transformer.window(), minFreq, *logger);
            auto internalAlgorithm = std::make_unique<PitchDetectorImpl>(
                std::move(transformer), std::move(autocorrPitchDetector), noisy.sampleRate, config,
                std::move(logger));
            PitchDetector* pitchDetector = internalAlgorithm.get();
            std::unique_ptr<PitchDetectorMedianFilter> medianFilter;

            if (!argTestWithMedianFilter.has_value() || *argTestWithMedianFilter) {
                medianFilter = std::make_unique<PitchDetectorMedianFilter>(
                    noisy.sampleRate, blockSize, std::move(internalAlgorithm));
                pitchDetector = medianFilter.get();
            }

            auto negativeCount = 0;
            auto falseNegativeCount = 0;
            auto positiveCount = 0;
            auto falsePositiveCount = 0;
            const auto numChannels = noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
            const auto numFrames = noisy.interleaved.size() / numChannels;
            const auto* noisyData = noisy.interleaved.data();

            std::vector<testUtils::ProcessEstimate> testFileEstimates;

            for (auto i = 0u; i + blockSize < numFrames; i += blockSize) {
                auto presenceScore = 0.f;
                auto result = pitchDetector->process(noisyData + i * numChannels, &presenceScore);
                const auto currentTime =
                    static_cast<double>(i + blockSize - pitchDetector->delaySamples()) /
                    noisy.sampleRate;
                const auto truth = (currentTime >= sample.truth.startTime) &&
                                   (currentTime <= sample.truth.endTime);
                if (truth) {
                    ++positiveCount;
                    if (result == 0.f)
                        ++falseNegativeCount;
                } else {
                    ++negativeCount;
                    if (result != 0.f)
                        ++falsePositiveCount;
                }
                const auto errorCents =
                    result > 0.f ? 1200.f * std::log2(result / sample.truth.frequency) : 0.f;
                testFileEstimates.emplace_back(truth, presenceScore, result, errorCents);
            }

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
            const auto outWavName = testUtils::getOutDir() / "wav" / (filename + ".wav");

            const auto displayCents = cents.value_or(testUtils::Cents{0.f, 0.f});
            const auto evalDir = testUtils::getEvalDir();
            std::stringstream csvLine;
            csvLine << displayCents.avg << "," << displayCents.rms << "," << FPR << "," << FNR
                    << "," << fs::relative(outWavName, evalDir) << "," << testCase.id << "\n";

            if (argTestCaseId.has_value()) {
                std::cout << csvLine.str();
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
        errorsFile << "errors = [";
        for (const auto& result : results) {
            if (result.cents.has_value()) {
                for (const auto& estimate : result.estimates) {
                    if (estimate.f > 0.f) {
                        errorsFile << estimate.e << ",";
                    }
                }
            }
        }
        errorsFile << "]";
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

    tee << "Error across all tests:\n\tAVG: " << avgAvg << "\n\tRMS: " << rmsAvg
        << "\n\tworst RMS error: " << worstRms << " at index " << worstRmsIndex << "\n";

    constexpr auto previousRmsError = 45.84872850647737;
    constexpr auto previousAuc = 0.902510855252703;

    constexpr auto comparisonTolerance = 0.01;
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

    {
        const auto thresholdFilePath =
            testUtils::getEvalDir() / "BenchmarkingOutput" / "threshold.txt";
        std::ofstream thresholdFile{thresholdFilePath};
        thresholdFile << rocInfo.threshold;
    }

    // If it changes and it's for the better, then it's probably a good thing, but
    // let's keep an eye on it anyway. If it's for the worse, then either there is
    // a good reason or we have a regression.
    EXPECT_TRUE(classifierQualityIsUnchanged)
        << "Classifier quality has changed! Previous AUC: " << previousAuc
        << ", new AUC: " << rocInfo.areaUnderCurve;
}
}  // namespace saint
