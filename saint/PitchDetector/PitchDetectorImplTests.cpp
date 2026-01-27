#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "DummyPitchDetectorLogger.h"
#include "PitchDetectorImpl.h"
#include "PitchDetectorLogger.h"
#include "testUtils.h"

namespace saint {

namespace fs = std::filesystem;

namespace {

struct RocResult {
    RocResult(bool t, double s) : truth(t), score(s) {}
    bool truth = false;
    double score = 0.0;
};

struct Noise {
    fs::path file;
    const char* rmsDb;
    std::vector<float> data;
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

std::vector<Noise> loadNoiseData() {
    std::vector<fs::path> noiseFiles;
    for (const auto& entry :
         fs::recursive_directory_iterator(testUtils::getEvalDir() / "testFiles" / "noise")) {
        if (entry.path().extension() == ".wav") {
            noiseFiles.push_back(entry.path());
        }
    }

    const std::vector<const char*> noiseRmsDb{"-40", "-30"};
    std::vector<Noise> noiseData;
    for (const auto& noiseFile : noiseFiles) {
        auto noiseAudio = testUtils::fromWavFile(noiseFile);
        if (!noiseAudio.has_value()) {
            continue;
        }
        for (const auto rmsDb : noiseRmsDb) {
            const float dB = std::stof(rmsDb);
            testUtils::scaleToRms(noiseAudio->data, dB);
            noiseData.push_back(Noise{noiseFile, rmsDb, noiseAudio->data});
        }
    }
    return noiseData;
}

// Targeting tuning of acoustic guitar:
// - min note accounts for a drop-D tuning and an additional tone to account for
// pitch changes while tuning
// - max note is the high E on the first string, adding a tone for margin
constexpr PitchDetector::Config config{
    Pitch{PitchClass::C, 2},
    Pitch{PitchClass::Gb, 4},
};
}  // namespace

TEST(PitchDetectorImpl, benchmarking) {
    const auto samples = loadSamples();
    const auto noiseData = loadNoiseData();

    std::vector<double> rmsErrors;
    std::vector<RocResult> rocResults;

    auto instanceCount = 0;
    std::optional<int> argInstanceCount;
    const auto& argv = ::testing::internal::GetArgvs();
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string argStr(argv[i]);
        const std::string prefix("instanceCount=");
        if (argStr.find(prefix) == 0) {
            argInstanceCount = std::stoi(argStr.substr(prefix.size()));
        }
    }

    for (auto s = 0; s < samples.size(); ++s) {
        const auto& sample = samples[s];
        const auto& testFile = sample.file;
        std::cout << "Processing " << testFile << "\n";

        const fs::path cleanFile = testUtils::getFileShortName(testFile);
        constexpr auto blockSize = 512;
        const std::optional<testUtils::Audio> clean = testUtils::fromWavFile(testFile);

        if (!clean.has_value()) {
            std::cerr << "Could not read file: " << testFile << "\n";
            continue;
        }

        for (auto n = 0; n < noiseData.size(); ++n) {
            testUtils::Finally incrementInstanceCount([&instanceCount]() { ++instanceCount; });

            if (argInstanceCount.has_value() && instanceCount != *argInstanceCount) {
                continue;
            }

            const auto& noise = noiseData[n];
            std::cout << "  Adding noise from " << noise.file << " at " << noise.rmsDb << " dB\n";
            auto noisy = clean->data;
            testUtils::mixNoise(noisy, noise.data);

            auto estimateIndex = 0;
            std::optional<int> logEstimateIndex;
            const std::string logPrefix("logEstimateIndex=");
            for (size_t i = 1; i < argv.size(); ++i) {
                const std::string argStr(argv[i]);
                if (argStr.find(logPrefix) == 0) {
                    logEstimateIndex = std::stoi(argStr.substr(logPrefix.size()));
                }
            }
            std::unique_ptr<PitchDetectorLoggerInterface> logger;
            if (logEstimateIndex.has_value()) {
                logger =
                    std::make_unique<PitchDetectorLogger>(clean->sampleRate, *logEstimateIndex);
            } else {
                logger = std::make_unique<DummyPitchDetectorLogger>();
            }
            const auto* loggerPtr = logger.get();

            PitchDetectorImpl sut(clean->sampleRate, config, std::move(logger));
            std::vector<float> results;
            for (auto i = 0u; i + blockSize < noisy.size(); i += blockSize) {
                auto presenceScore = 0.f;
                auto result = sut.process(noisy.data() + i, blockSize, &presenceScore);
                const auto currentTime = static_cast<double>(i + blockSize) / clean->sampleRate;
                const auto truth = (currentTime >= sample.truth.startTime) &&
                                   (currentTime <= sample.truth.endTime);
                if (result.has_value()) {
                    ++estimateIndex;
                    results.push_back(*result);
                    rocResults.emplace_back(truth, presenceScore);
                }
            }

            const auto filename = cleanFile.string() + "_with_" + noise.file.stem().string() +
                                  "_at_" + noise.rmsDb + "dB";
            const auto outWavName = testUtils::getOutDir() / "wav" / (filename + ".wav");
            testUtils::toWavFile(outWavName, {noisy, clean->sampleRate});

            const auto resultPath = testUtils::getOutDir() / (cleanFile.string() + "_results.py");
            const auto rmsError = testUtils::writeResultFile(sample, results, resultPath);
            rmsErrors.push_back(rmsError);

            if (auto realLogger = dynamic_cast<PitchDetectorLogger const*>(loggerPtr);
                realLogger && realLogger->analysisAudioIndex()) {
                const auto endIndex = *realLogger->analysisAudioIndex();
                const auto startIndex = endIndex - sut.windowSizeSamples();
                const testUtils::Marking marking{startIndex, endIndex};
                testUtils::writeMarkedWavFile(filename, noisy, clean->sampleRate, marking);
            }

            std::cout << "    RMS error: " << rmsError << " cents, instanceCount: " << instanceCount
                      << "\n";
        }
    }

    auto rmsAvg = 0.;
    for (const auto e : rmsErrors) {
        rmsAvg += e;
    }
    if (!rmsErrors.empty()) {
        rmsAvg /= static_cast<double>(rmsErrors.size());
    }
    std::cout << "Average RMS error across all tests: " << rmsAvg << " cents\n";

    constexpr auto previousRmsError = 178.02243990022137;
    constexpr auto comparisonTolerance = 0.01;
    const auto rmsErrorIsUnchanged = testUtils::valueIsUnchanged(
        testUtils::getEvalDir() / "BenchmarkingOutput" / "RMS_error.txt", previousRmsError, rmsAvg,
        comparisonTolerance);
    EXPECT_TRUE(rmsErrorIsUnchanged)
        << "RMS error has changed! Previous RMS error: " << previousRmsError
        << ", new RMS error: " << rmsAvg;

    constexpr auto allowedFalsePositiveRate = 0.05;
    const testUtils::RocInfo rocInfo =
        testUtils::GetRocInfo<RocResult>(rocResults, allowedFalsePositiveRate);

    std::ofstream rocFile(testUtils::getOutDir() / "roc_curve.py");
    rocFile << "AUC = " << rocInfo.areaUnderCurve << "\n";
    rocFile << "threshold = " << rocInfo.threshold << "\n";
    rocFile << "allowedFalsePositiveRate = " << allowedFalsePositiveRate << "\n";
    testUtils::PrintPythonVector(rocFile, rocInfo.falsePositiveRates, "falsePositiveRates");
    testUtils::PrintPythonVector(rocFile, rocInfo.truePositiveRates, "truePositiveRates");

    constexpr auto previousAuc = 0.89471154289243526;
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
