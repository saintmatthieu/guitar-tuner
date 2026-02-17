#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "PitchDetectorTypes.h"
#include "PitchDetectorUtils.h"
#include "testUtils.h"

namespace saint {

namespace fs = std::filesystem;

// Create a test case ID from clean file, noise file, and noise level.
// Format: cleanFilePath|noiseFilePath|dBLevel (using relative paths from eval dir)
inline std::string computeTestCaseId(const fs::path& cleanFile, const fs::path& noiseFile,
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

inline std::optional<ParsedTestCaseId> parseTestCaseId(const std::string& id) {
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

inline std::vector<testUtils::Sample> loadSamples() {
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

inline std::vector<Noise> loadNoiseData(int numFrames, const fs::path& silenceFilePath) {
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
constexpr PitchDetectorConfig kTestConfig{
    Pitch{PitchClass::Db, 2},
    Pitch{PitchClass::Gb, 4},
};

// Create a single test case from a parsed ID
inline std::optional<TestCase> createTestCaseFromId(const std::string& testCaseId) {
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

inline std::vector<TestCase> prepareTestCases(const std::optional<std::string>& argTestCaseId) {
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

}  // namespace saint
