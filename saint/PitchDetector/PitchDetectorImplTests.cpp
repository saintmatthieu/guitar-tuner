#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>

#if defined(__APPLE__)
  #include <tbb/parallel_for.h>
  #include <tbb/blocked_range.h>
#else
  #include <execution>
#endif

#include "DummyPitchDetectorLogger.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorLogger.h"
#include "PitchDetectorMedianFilter.h"
#include "Utils.h"
#include "testUtils.h"

namespace saint {

namespace fs = std::filesystem;

namespace {
struct Noise {
    fs::path file;
    const char* rmsDb;
    std::vector<float> data;
};

struct TestCase {
    int index;
    testUtils::Sample sample;
    Noise noise;
    testUtils::Audio noisy;
    int blockSize;
};

struct TestResult {
    int index;
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
    return samples;
}

std::vector<Noise> loadNoiseData(int numSamples, const fs::path& silenceFilePath) {
    std::vector<fs::path> noiseFiles;
    for (const auto& entry :
         fs::recursive_directory_iterator(testUtils::getEvalDir() / "testFiles" / "noise")) {
        if (entry.path().extension() == ".wav") {
            noiseFiles.push_back(entry.path());
        }
    }

    std::vector<Noise> noiseData;

    const auto silenceAudio = testUtils::fromWavFile(silenceFilePath, numSamples);
    noiseData.push_back(Noise{silenceFilePath, "-inf", silenceAudio->interleaved});

    const std::vector<const char*> noiseRmsDb{"-40", "-50", "-60"};
    for (const auto& noiseFile : noiseFiles) {
        auto noiseAudio = testUtils::fromWavFile(noiseFile, numSamples);
        if (!noiseAudio.has_value()) {
            continue;
        }
        for (const auto rmsDb : noiseRmsDb) {
            const float dB = std::stof(rmsDb);
            testUtils::scaleToRms(noiseAudio->interleaved, dB);
            noiseData.push_back(Noise{noiseFile, rmsDb, noiseAudio->interleaved});
        }
    }

    return noiseData;
}

// Targeting tuning of acoustic guitar:
// - min note accounts for a drop-D tuning and an additional tone to account for
// pitch changes while tuning
// - max note is the high E on the first string, adding a tone for margin
constexpr PitchDetector::Config config{
    Pitch{PitchClass::Db, 2},
    Pitch{PitchClass::Gb, 4},
};

std::vector<TestCase> prepareTestCases(const std::optional<fs::path>& argSampleFile,
                                       const std::optional<int>& argInstanceCount) {
    const auto samples = loadSamples();

    constexpr auto forceWriteFiles = false;
    auto silenceWriter = std::make_shared<testUtils::RealFileWriter>();
    std::shared_ptr<testUtils::FileWriter> fileWriter;
    if (forceWriteFiles || argInstanceCount.has_value() || argSampleFile.has_value()) {
        fileWriter = silenceWriter;
    } else {
        fileWriter = std::make_shared<testUtils::DummyFileWriter>();
    }

    const std::vector<float> silence(44100, 0.f);
    const auto silenceFilePath = testUtils::getOutDir() / "wav" / "silence.wav";
    silenceWriter->toWavFile(silenceFilePath, {silence, 44100, ChannelFormat::Mono}, nullptr);

    std::vector<TestCase> testCases;
    int instanceCount = 0;

    std::cout << "Preparing test cases..." << std::endl;
    auto testCaseCount = 0;
    for (const auto& sample : samples) {
        std::cout << "\r" << ++testCaseCount << "/" << samples.size() << std::flush;
        const auto& testFile = sample.file;

        std::optional<testUtils::Audio> clean = testUtils::fromWavFile(testFile);
        if (!clean.has_value()) {
            std::cerr << "Could not read file: " << testFile << "\n";
            continue;
        }

        const auto blockSize = clean->sampleRate / 100;
        testUtils::scaleToPeak(clean->interleaved, -10.f);

        const auto noiseData = loadNoiseData(clean->interleaved.size(), silenceFilePath);

        const auto takeSample =
            !argSampleFile.has_value() ||
            (fs::exists(*argSampleFile) && fs::equivalent(*argSampleFile, testFile));

        for (const auto& noise : noiseData) {
            const auto takeTestCase =
                takeSample && (!argInstanceCount.has_value() || instanceCount == *argInstanceCount);

            if (takeTestCase) {
                auto noisy = *clean;
                testUtils::mixNoise(noisy, noise.data);
                testCases.push_back(
                    TestCase{instanceCount, sample, noise, std::move(noisy), blockSize});
            }
            ++instanceCount;
        }
    }
    return testCases;
}
}  // namespace

TEST(PitchDetectorImpl, benchmarking) {
    std::cout << "\n";

    const auto logFilePath = testUtils::getOutDir() / "benchmarking.log";
    std::ofstream logFile(logFilePath);
    testUtils::TeeStream tee(std::cout, logFile);

    std::optional<int> argInstanceCount;
    const auto& argv = ::testing::internal::GetArgvs();
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string argStr(argv[i]);
        const std::string prefix("instanceCount=");
        if (argStr.find(prefix) == 0) {
            argInstanceCount = std::stoi(argStr.substr(prefix.size()));
        }
    }

    std::optional<fs::path> argSampleFile;
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string argStr(argv[i]);
        const std::string prefix("sample=");
        if (argStr.find(prefix) == 0) {
            argSampleFile = fs::path(argStr.substr(prefix.size()));
        }
    }

    std::optional<std::ofstream> csvFile;

    if (!argInstanceCount.has_value() && !argSampleFile.has_value()) {
        const auto csvFilePath = testUtils::getOutDir() / "benchmarking.csv";
        csvFile.emplace(csvFilePath);
        *csvFile << "index,AVG,RMS,FPR,FNR,testFile,noiseFile,noiseDb,mix\n";
    }

    // Build all test cases upfront
    const std::vector<TestCase> testCases = prepareTestCases(argSampleFile, argInstanceCount);

    const auto numEvaluations = testCases.size();

    // Pre-allocate results vector for thread-safe indexed access
    std::vector<TestResult> results(testCases.size());
    std::atomic<int> completedCount{0};
    std::mutex progressMutex;

    std::cout << std::endl << "Evaluating samples..." << std::endl;
    
    auto processOne = [&](size_t idx) {
        const auto& testCase = testCases[idx];
        const auto& sample = testCase.sample;
        const auto& noisy = testCase.noisy;
        const auto blockSize = testCase.blockSize;

        auto logger = std::make_unique<DummyPitchDetectorLogger>();

        auto internalAlgorithm = std::make_unique<PitchDetectorImpl>(
            noisy.sampleRate, noisy.channelFormat, blockSize, config, std::move(logger));
        PitchDetectorMedianFilter sut(noisy.sampleRate, blockSize,
                                      std::move(internalAlgorithm));

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
            auto result = sut.process(noisyData + i * numChannels, &presenceScore);
            const auto currentTime =
                static_cast<double>(i + blockSize - sut.delaySamples()) / noisy.sampleRate;
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

        const std::optional<testUtils::Cents> cents =
            testUtils::getError(sample, testFileEstimates);

        const fs::path cleanFile = testUtils::getFileShortName(sample.file);
        const auto filename = cleanFile.string() + "_with_" +
                              testCase.noise.file.stem().string() + "_at_" +
                              testCase.noise.rmsDb + "dB";
        const auto outWavName = testUtils::getOutDir() / "wav" / (filename + ".wav");

        const auto displayCents = cents.value_or(testUtils::Cents{0.f, 0.f});
        const auto evalDir = testUtils::getEvalDir();

        std::stringstream csvLine;
        csvLine << testCase.index << "," << displayCents.avg << "," << displayCents.rms << ","
                << FPR << "," << FNR << "," << fs::relative(sample.file, evalDir) << ","
                << fs::relative(testCase.noise.file, evalDir) << "," << testCase.noise.rmsDb
                << "," << fs::relative(outWavName, evalDir) << "\n";

        results[idx] = TestResult{
            testCase.index,
            std::move(testFileEstimates),
            cents,
            FPR,
            FNR,
            csvLine.str(),
            sample.file,
            testCase.noise.file,
            testCase.noise.rmsDb
        };

        const auto completed = ++completedCount;
        {
            std::lock_guard<std::mutex> lock(progressMutex);
            std::cout << "\r" << completed << "/" << numEvaluations << std::flush;
        }
    };

    // Process test cases in parallel
    #if defined(__APPLE__)
    // macOS: oneTBB
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, testCases.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                processOne(i);
            }
        });
    #else
    // Windows / Linux: std::execution::par
    std::for_each(
        std::execution::par,
        size_t{0}, testCases.size(),
        [&](size_t i) {
            processOne(i);
        });
    #endif

    std::cout << "\n";

    // Aggregate results (sequential)
    std::vector<testUtils::ProcessEstimate> estimatesForRoc;
    std::vector<std::optional<testUtils::Cents>> allTestFileEstimates;

    // Sort results by index to maintain deterministic output order
    std::sort(results.begin(), results.end(),
              [](const TestResult& a, const TestResult& b) { return a.index < b.index; });

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

    if (argInstanceCount.has_value() || argSampleFile.has_value()) {
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

    constexpr auto previousRmsError = 64.04810566642929;
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
