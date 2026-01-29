#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

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

    const std::vector<const char*> noiseRmsDb{"-40", "-30"};
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
    Pitch{PitchClass::G, 1},
    Pitch{PitchClass::A, 4},
};
}  // namespace

TEST(PitchDetectorImpl, benchmarking) {
    std::cout << "\n";

    std::vector<testUtils::Result> results;
    std::vector<double> rmsErrors;

    const auto logFilePath = testUtils::getOutDir() / "benchmarking.log";
    std::ofstream logFile(logFilePath);
    testUtils::TeeStream tee(std::cout, logFile);

    const auto samples = loadSamples();

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

    const std::vector<float> silence(44100, 0.f);
    const auto silenceFilePath = testUtils::getOutDir() / "wav" / "silence.wav";
    testUtils::toWavFile(silenceFilePath, {silence, 44100, ChannelFormat::Mono});

    const auto numSamples = samples.size();
    const auto numNoises =
        loadNoiseData(1000 /*arbitrary, small number of samples*/, silenceFilePath).size();
    const auto numEvaluations = numSamples * numNoises;

    for (auto s = 0; s < samples.size(); ++s) {
        const testUtils::Sample& sample = samples[s];
        const auto& testFile = sample.file;

        const fs::path cleanFile = testUtils::getFileShortName(testFile);
        const std::optional<testUtils::Audio> clean = testUtils::fromWavFile(testFile);
        const auto blockSize = clean->sampleRate / 100;  // To make debugging easier

        if (!clean.has_value()) {
            std::cerr << "Could not read file: " << testFile << "\n";
            continue;
        }

        const auto noiseData = loadNoiseData(clean->interleaved.size(), silenceFilePath);

        auto somethingProcessed = false;
        for (auto n = 0; n < noiseData.size(); ++n) {
            const auto takeTestCase =
                !argInstanceCount.has_value() || instanceCount == *argInstanceCount;
            utils::Finally incrementInstanceCount([&instanceCount, takeTestCase, &tee]() {
                ++instanceCount;
                if (takeTestCase)
                    tee << "\n";
            });

            if (!takeTestCase) {
                continue;
            } else {
                somethingProcessed = true;
            }

            const Noise& noise = noiseData[n];
            const auto shortFileName =
                noise.file.parent_path().stem() /
                (std::to_string(instanceCount) + "_" + noise.file.stem().string());
            const auto noiseFileName =
                testUtils::getOutDir() / "wav" /
                (shortFileName.string() + "_" + noise.rmsDb + "dB_" + ".wav");
            testUtils::toWavFile(noiseFileName,
                                 {noise.data, clean->sampleRate, clean->channelFormat}, "NOISE");

            auto noisy = *clean;
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

            auto internalAlgorithm = std::make_unique<PitchDetectorImpl>(
                clean->sampleRate, clean->channelFormat, blockSize, config, std::move(logger));
            const auto windowSizeSamples = internalAlgorithm->windowSizeSamples();
            PitchDetectorMedianFilter sut(clean->sampleRate, blockSize,
                                          std::move(internalAlgorithm));
            auto negativeCount = 0;
            auto falseNegativeCount = 0;
            auto positiveCount = 0;
            auto falsePositiveCount = 0;
            const auto numChannels = clean->channelFormat == ChannelFormat::Mono ? 1 : 2;
            const auto numFrames = noisy.interleaved.size() / numChannels;
            const auto* noisyData = noisy.interleaved.data();

            std::vector<testUtils::Result> sampleResults;

            std::vector<float> presenceScoreAsAudio;
            if (logEstimateIndex.has_value()) {
                presenceScoreAsAudio.resize(numFrames);
            }

            for (auto i = 0u; i + blockSize < numFrames; i += blockSize) {
                auto presenceScore = 0.f;
                std::vector<float> frame(noisyData + i * numChannels,
                                         noisyData + (i + blockSize) * numChannels);
                auto result = sut.process(noisyData + i * numChannels, &presenceScore);
                const auto currentTime = static_cast<double>(i + blockSize) / clean->sampleRate;
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
                ++estimateIndex;
                sampleResults.emplace_back(truth, presenceScore, result, currentTime);
                if (logEstimateIndex.has_value()) {
                    std::fill(presenceScoreAsAudio.begin() + i,
                              presenceScoreAsAudio.begin() + i + blockSize, presenceScore);
                }
            }

            if (logEstimateIndex.has_value()) {
                const auto presenceScoreWavName = testUtils::getOutDir() / "presenceScore.wav";
                testUtils::toWavFile(presenceScoreWavName,
                                     {presenceScoreAsAudio, clean->sampleRate, ChannelFormat::Mono},
                                     "PRESENCE");
            }

            const auto FPR = 1. * falsePositiveCount / negativeCount;
            const auto FNR = 1. * falseNegativeCount / positiveCount;

            const auto filename = cleanFile.string() + "_with_" + noise.file.stem().string() +
                                  "_at_" + noise.rmsDb + "dB";
            const auto outWavName = testUtils::getOutDir() / "wav" / (filename + ".wav");
            testUtils::toWavFile(outWavName, noisy, "MIX");

            const auto resultPath = testUtils::getOutDir() / (cleanFile.string() + "_results.py");
            const auto rmsError = testUtils::writeResultFile(sample, sampleResults, resultPath);

            rmsErrors.push_back(rmsError);
            results.insert(results.end(), sampleResults.begin(), sampleResults.end());

            if (auto realLogger = dynamic_cast<PitchDetectorLogger const*>(loggerPtr);
                realLogger && realLogger->analysisAudioIndex()) {
                const auto startIndex = *realLogger->analysisAudioIndex();
                const auto endIndex = startIndex + windowSizeSamples;
                const testUtils::Marking marking{startIndex, endIndex};
                testUtils::writeMarkedWavFile(filename, clean->sampleRate, clean->numFrames(),
                                              marking);
            }

            tee << "STATS\t" << instanceCount << "/" << numEvaluations
                << ": RMS error: " << rmsError << " cents, FPR: " << FPR << ", FNR: " << FNR
                << "\n";
        }

        if (somethingProcessed)
            tee << "Finished with " << testFile << "\n\n";
    }

    if (argInstanceCount.has_value()) {
        return;
    }

    auto rmsAvg = 0.;
    for (const auto e : rmsErrors) {
        rmsAvg += e;
    }

    rmsAvg /= static_cast<double>(rmsErrors.size());
    const auto worstRmsIt = std::max_element(rmsErrors.begin(), rmsErrors.end());
    const auto worstRmsIndex = std::distance(rmsErrors.begin(), worstRmsIt);
    tee << "Average RMS error across all tests: " << rmsAvg
        << " cents, worst RMS error: " << *worstRmsIt << " at index " << worstRmsIndex << "\n";

    constexpr auto previousRmsError = 11.76702707717938;
    constexpr auto previousAuc = 0.9211360705408158;

    constexpr auto comparisonTolerance = 0.01;
    const auto rmsErrorIsUnchanged = testUtils::valueIsUnchanged(
        testUtils::getEvalDir() / "BenchmarkingOutput" / "RMS_error.txt", previousRmsError, rmsAvg,
        comparisonTolerance);
    EXPECT_TRUE(rmsErrorIsUnchanged)
        << "RMS error has changed! Previous RMS error: " << previousRmsError
        << ", new RMS error: " << rmsAvg;

    constexpr auto allowedFalsePositiveRate = 0.01;  // 1%
    const testUtils::RocInfo rocInfo =
        testUtils::GetRocInfo<testUtils::Result>(results, allowedFalsePositiveRate);

    std::ofstream rocFile(testUtils::getOutDir() / "roc_curve.py");
    rocFile << "AUC = " << rocInfo.areaUnderCurve << "\n";
    rocFile << "threshold = " << rocInfo.threshold << "\n";
    rocFile << "allowedFalsePositiveRate = " << allowedFalsePositiveRate << "\n";
    testUtils::PrintPythonVector(rocFile, rocInfo.falsePositiveRates, "falsePositiveRates");
    testUtils::PrintPythonVector(rocFile, rocInfo.truePositiveRates, "truePositiveRates");

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
