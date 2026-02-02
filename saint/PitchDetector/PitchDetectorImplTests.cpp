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
}  // namespace

TEST(PitchDetectorImpl, benchmarking) {
    std::cout << "\n";

    std::vector<testUtils::ProcessEstimate> estimatesForRoc;
    std::vector<std::optional<testUtils::Cents>> allTestFileEstimates;

    const auto logFilePath = testUtils::getOutDir() / "benchmarking.log";
    std::ofstream logFile(logFilePath);
    testUtils::TeeStream tee(std::cout, logFile);

    const auto csvFilePath = testUtils::getOutDir() / "benchmarking.csv";
    std::ofstream csvFile(csvFilePath);
    csvFile << "index,AVG,RMS,FPR,FNR,testFile,noiseFile,noiseDb,mix\n";

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

    std::optional<fs::path> argSampleFile;
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string argStr(argv[i]);
        const std::string prefix("sample=");
        if (argStr.find(prefix) == 0) {
            argSampleFile = fs::path(argStr.substr(prefix.size()));
        }
    }

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

    const auto numSamples = samples.size();
    const auto numNoises =
        loadNoiseData(1000 /*arbitrary, small number of samples*/, silenceFilePath).size();
    const auto numEvaluations = numSamples * numNoises;

    for (auto s = 0; s < samples.size(); ++s) {
        const testUtils::Sample& sample = samples[s];
        const auto& testFile = sample.file;

        const fs::path cleanFile = testUtils::getFileShortName(testFile);
        std::optional<testUtils::Audio> clean = testUtils::fromWavFile(testFile);
        const auto blockSize = clean->sampleRate / 100;  // To make debugging easier

        if (!clean.has_value()) {
            std::cerr << "Could not read file: " << testFile << "\n";
            continue;
        }

        testUtils::scaleToPeak(clean->interleaved, -10.f);

        const auto noiseData = loadNoiseData(clean->interleaved.size(), silenceFilePath);

        const auto takeSample =
            !argSampleFile.has_value() ||
            (fs::exists(*argSampleFile) && fs::equivalent(*argSampleFile, testFile));

        auto somethingProcessed = false;
        for (auto n = 0; n < noiseData.size(); ++n) {
            const auto takeTestCase =
                takeSample && (!argInstanceCount.has_value() || instanceCount == *argInstanceCount);
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
            fileWriter->toWavFile(noiseFileName,
                                  {noise.data, clean->sampleRate, clean->channelFormat}, &tee,
                                  "NOISE");

            auto noisy = *clean;
            testUtils::mixNoise(noisy, noise.data);

            auto processIndex = 0;
            std::optional<int> indexOfProcessToLog;
            const std::string logPrefix("indexOfProcessToLog=");
            for (size_t i = 1; i < argv.size(); ++i) {
                const std::string argStr(argv[i]);
                if (argStr.find(logPrefix) == 0) {
                    indexOfProcessToLog = std::stoi(argStr.substr(logPrefix.size()));
                }
            }
            std::unique_ptr<PitchDetectorLoggerInterface> logger;
            if (indexOfProcessToLog.has_value()) {
                logger =
                    std::make_unique<PitchDetectorLogger>(clean->sampleRate, *indexOfProcessToLog);
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

            std::vector<testUtils::ProcessEstimate> testFileEstimates;

            std::vector<float> presenceScoreAsAudio;
            if (indexOfProcessToLog.has_value()) {
                presenceScoreAsAudio.resize(numFrames);
            }

            for (auto i = 0u; i + blockSize < numFrames; i += blockSize) {
                auto presenceScore = 0.f;
                auto unfilteredEstimate = 0.f;
                std::vector<float> frame(noisyData + i * numChannels,
                                         noisyData + (i + blockSize) * numChannels);
                auto result =
                    sut.process(noisyData + i * numChannels, &presenceScore, &unfilteredEstimate);
                const auto currentTime =
                    static_cast<double>(i + blockSize - sut.delaySamples()) / clean->sampleRate;
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
                ++processIndex;
                testFileEstimates.emplace_back(truth, presenceScore, result, unfilteredEstimate);
                if (indexOfProcessToLog.has_value()) {
                    std::fill(presenceScoreAsAudio.begin() + i,
                              presenceScoreAsAudio.begin() + i + blockSize, presenceScore);
                }
            }

            if (indexOfProcessToLog.has_value()) {
                const auto presenceScoreWavName = testUtils::getOutDir() / "presenceScore.wav";
                fileWriter->toWavFile(
                    presenceScoreWavName,
                    {presenceScoreAsAudio, clean->sampleRate, ChannelFormat::Mono}, &tee,
                    "PRESENCE");
            }

            const auto FPR = 1. * falsePositiveCount / negativeCount;
            const auto FNR = 1. * falseNegativeCount / positiveCount;

            const auto filename = cleanFile.string() + "_with_" + noise.file.stem().string() +
                                  "_at_" + noise.rmsDb + "dB";
            const auto outWavName = testUtils::getOutDir() / "wav" / (filename + ".wav");
            fileWriter->toWavFile(outWavName, noisy, &tee, "MIX");

            const std::optional<testUtils::Cents> cents =
                testUtils::getError(sample, testFileEstimates);
            allTestFileEstimates.push_back(cents);

            estimatesForRoc.insert(estimatesForRoc.end(), testFileEstimates.begin(),
                                   testFileEstimates.end());

            if (auto realLogger = dynamic_cast<PitchDetectorLogger const*>(loggerPtr);
                realLogger && realLogger->analysisAudioIndex()) {
                const auto startIndex = *realLogger->analysisAudioIndex();
                const auto endIndex = startIndex + windowSizeSamples;
                const testUtils::Marking marking{startIndex, endIndex};
                testUtils::writeLogMarks(filename, clean->sampleRate, marking);
            }

            const auto displayCents = cents.value_or(testUtils::Cents{0.f, 0.f});
            const auto evalDir = testUtils::getEvalDir();
            std::stringstream csvLine;
            csvLine << instanceCount << "," << displayCents.avg << "," << displayCents.rms << ","
                    << FPR << "," << FNR << "," << fs::relative(testFile, evalDir) << ","
                    << fs::relative(noise.file, evalDir) << "," << noise.rmsDb << ","
                    << fs::relative(outWavName, evalDir) << "\n";
            csvFile << csvLine.str();

            std::cout << instanceCount << "/" << numEvaluations;
        }

        if (somethingProcessed)
            tee << "Finished with " << testFile << "\n\n";
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

    constexpr auto previousRmsError = 76.62722630301519;
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
