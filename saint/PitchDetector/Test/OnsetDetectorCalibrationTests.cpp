#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "OnsetDetector.h"
#include "TestCaseUtils.h"
#include "Utils.h"
#include "testUtils.h"

namespace saint {

namespace fs = std::filesystem;

namespace {

constexpr double kDelaySeconds = 1.;

// Mix the clean signal with itself delayed by `delaySamples` to simulate
// plucking a string while the previous pluck is still ringing.
void mixWithDelayedSelf(std::vector<float>& data, int delaySamples) {
    for (size_t i = delaySamples; i < data.size(); ++i) {
        data[i] += data[i - delaySamples];
    }
}

// Prepare test cases with overlapping plucks: noisy = clean + clean_delayed + noise.
std::vector<TestCase> prepareOnsetTestCases(const std::optional<std::string>& argTestCaseId) {
    // Use the same preparation as pitch detector tests, then overlay with delayed self.
    auto testCases = prepareTestCases(argTestCaseId);

    for (auto& tc : testCases) {
        const auto numChannels = tc.noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
        const int delaySamples = (tc.noisy.sampleRate * kDelaySeconds) * numChannels;

        // We need the original clean signal to create the delayed version.
        // Reload it from the sample file.
        auto clean = testUtils::fromWavFile(tc.sample.file);
        if (!clean.has_value())
            continue;
        testUtils::scaleToPeak(clean->interleaved, -10.f);

        // Pad clean to match noisy length if needed, then add delayed copy into noisy.
        auto& cleanData = clean->interleaved;
        cleanData.resize(tc.noisy.interleaved.size(), 0.f);

        // Add the delayed clean signal onto the noisy mix.
        for (size_t i = delaySamples; i < tc.noisy.interleaved.size(); ++i) {
            tc.noisy.interleaved[i] += cleanData[i - delaySamples];
        }
    }

    return testCases;
}

std::vector<float> extractDebugOutput(const std::vector<DebugOutput>& debugOutputs,
                                      const std::string& key) {
    std::vector<float> values;
    values.reserve(debugOutputs.size());
    for (const auto& debugOutput : debugOutputs) {
        auto it = debugOutput.find(key);
        if (it != debugOutput.end()) {
            values.push_back(it->second);
        } else {
            values.push_back(0.f);
        }
    }
    return values;
}

void writeToWavFile(const std::vector<DebugOutput>& debugOutputs, const std::string& key,
                    int sampleRate, int blockSize, testUtils::TeeStream* tee) {
    auto values = extractDebugOutput(debugOutputs, key);
    const auto scaledValues = testUtils::scaleByPowerOf10(values);
    const auto filePath = testUtils::getOutDir() / (key + ".wav");
    testUtils::toWavFile(
        filePath, testUtils::Audio{std::move(scaledValues), sampleRate / blockSize}, tee, key);
}

void writeToWavFile(const std::vector<DebugOutput>& debugOutputs, int sampleRate, int blockSize,
                    testUtils::TeeStream* tee) {
    std::unordered_set<std::string> keys;
    for (const auto& debugOutput : debugOutputs) {
        for (const auto& [key, _] : debugOutput) {
            keys.insert(key);
        }
    }
    for (const auto& key : keys) {
        writeToWavFile(debugOutputs, key, sampleRate, blockSize, tee);
    }
}
}  // namespace

TEST(OnsetDetector, calibration) {
    std::cout << "\n";

    const auto minFreq = getMinFreq(kTestConfig);

    const auto logFilePath = testUtils::getOutDir() / "onset_calibration.log";
    std::ofstream logFile(logFilePath);
    testUtils::TeeStream tee(std::cout, logFile);

    const auto argTestCaseId = getArgument<std::string>("testCaseId");

    const std::vector<TestCase> testCases = prepareOnsetTestCases(argTestCaseId);
    const auto numEvaluations = testCases.size();

    std::atomic<int> completedCount{0};
    std::mutex progressMutex;

    tee << "Evaluating onset detection on " << numEvaluations << " test cases...\n";

    std::vector<float> onsetValues;
    std::vector<float> nonOnsetValues;

    // clang-format off
    const std::vector<std::string> blacklist{
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/D3.wav | testFiles/noise/kitchen_noise.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/B3.wav | testFiles/noise/kitchen_noise.wav | -40",
        "testFiles/notes/Fender_(Acoustic)/iPhone_17/D3.wav | testFiles/noise/kitchen_noise.wav | -40",
        "testFiles/notes/Fender_(Acoustic)/iPhone_17/D3.wav | testFiles/noise/CS_Telecaster_noise.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/E4.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/E4.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/E2.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/E2.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/E2.wav | testFiles/noise/home_4.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/E2.wav | testFiles/noise/kitchen_noise.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/B3.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/G3.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/B3.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/G3.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/G3.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/B3.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/G3.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/G3.wav | testFiles/noise/home_4.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/B3.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/B3.wav | testFiles/noise/home_4.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/D3.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/D3.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Fender_Stratocaster_unplugged/iPhone_11/D3.wav | testFiles/noise/home_4.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/D3.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/D3.wav | testFiles/noise/home_2.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/D3.wav | testFiles/noise/home_3.wav | -40",
        "testFiles/notes/Strandberg_unplugged/Nothing_Phone_2a/D3.wav | testFiles/noise/home_4.wav | -40",
        "testFiles/notes/Fender_(Acoustic)/iPhone_17/D3.wav | testFiles/noise/home_1.wav | -40",
        "testFiles/notes/Fender_(Acoustic)/iPhone_17/D3.wav | testFiles/noise/home_2.wav | -40",
        "testFiles/notes/Fender_(Acoustic)/iPhone_17/D3.wav | testFiles/noise/home_3.wav | -40",
    };
    // clang-format on

    auto processTestCases = [&](size_t startIdx, size_t endIdx) {
        for (size_t idx = startIdx; idx < endIdx; ++idx) {
            const auto& testCase = testCases[idx];
            const auto& noisy = testCase.noisy;
            const auto blockSize = testCase.blockSize;

            OnsetDetector onsetDetector(noisy.sampleRate, noisy.channelFormat, blockSize, minFreq);

            const auto numChannels = noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
            const auto numFrames = noisy.interleaved.size() / numChannels;

            std::vector<DebugOutput> debugOutputs;

            auto filteredNoisy = noisy.interleaved;
            float* const noisyData = filteredNoisy.data();

            for (size_t i = 0; i + blockSize < numFrames; i += blockSize) {
                DebugOutput debugOutput;
                onsetDetector.process(noisyData + i * numChannels, &debugOutput);
                debugOutputs.push_back(std::move(debugOutput));
            }

            const auto blocksPerSecond = 1.f * noisy.sampleRate / blockSize;

            const auto firstOnsetTime = testCase.sample.truth.startTime;  // as per labels
            const auto secondOnsetTime = firstOnsetTime + kDelaySeconds;
            const int firstOnsetBlockIndex = firstOnsetTime * blocksPerSecond;
            const int secondOnsetBlockIndex = secondOnsetTime * blocksPerSecond;
            const auto marginBeforeSeconds = 0.05f;
            const auto marginAfterSeconds = 0.15f;
            const int marginBeforeBlocks = marginBeforeSeconds * blocksPerSecond + .5f;
            const int marginAfterBlocks = marginAfterSeconds * blocksPerSecond + .5f;

            const auto onsetStrengths = extractDebugOutput(debugOutputs, "onsetStrength");
            const auto a = onsetStrengths.begin() +
                           std::max<int>(firstOnsetBlockIndex - marginBeforeBlocks, 0);
            const auto b = onsetStrengths.begin() + firstOnsetBlockIndex + marginAfterBlocks;
            const auto c = onsetStrengths.begin() + secondOnsetBlockIndex - marginBeforeBlocks;
            const auto d =
                onsetStrengths.begin() +
                std::min<int>(secondOnsetBlockIndex + marginAfterBlocks, onsetStrengths.size());

            const auto firstMaxIt = std::max_element(a, b);
            const auto firstMax = *firstMaxIt;
            const auto secondMaxIt = std::max_element(c, d);
            const auto secondMax = *secondMaxIt;

            if (argTestCaseId == testCase.id) {
                writeToWavFile(debugOutputs, noisy.sampleRate, blockSize, &tee);

                testUtils::toWavFile(
                    testUtils::getOutDir() / "noisy.wav",
                    testUtils::Audio{noisy.interleaved, noisy.sampleRate, noisy.channelFormat},
                    &tee, "input signal");

                testUtils::toWavFile(
                    testUtils::getOutDir() / "filteredNoisy.wav",
                    testUtils::Audio{filteredNoisy, noisy.sampleRate, noisy.channelFormat}, &tee,
                    "high-pass filtered input signal");

                std::ofstream labels(testUtils::getOutDir() / "filteredNoisy.txt");
                labels << firstOnsetTime - marginBeforeSeconds << "\t"
                       << firstOnsetTime + marginAfterSeconds << std::endl;
                labels << secondOnsetTime - marginBeforeSeconds << "\t"
                       << secondOnsetTime + marginAfterSeconds << std::endl;
            }

            const auto completed = ++completedCount;
            {
                std::lock_guard<std::mutex> lock(progressMutex);

                onsetValues.push_back(firstMax);
                onsetValues.push_back(secondMax);

                std::vector<float> newNonOnsetValues;
                newNonOnsetValues.insert(newNonOnsetValues.end(), onsetStrengths.begin(), a);
                newNonOnsetValues.insert(newNonOnsetValues.end(), b, c);
                newNonOnsetValues.insert(newNonOnsetValues.end(), d, onsetStrengths.end());

                const auto leastOfMax = std::min(firstMax, secondMax);
                const auto inBlacklist =
                    std::find(blacklist.begin(), blacklist.end(), testCase.id) != blacklist.end();
                const auto addWarning =
                    !inBlacklist && std::any_of(newNonOnsetValues.begin(), newNonOnsetValues.end(),
                                                [leastOfMax](float nonOnsetEstimate) {
                                                    return nonOnsetEstimate > leastOfMax;
                                                });

                nonOnsetValues.insert(nonOnsetValues.end(), newNonOnsetValues.begin(),
                                      newNonOnsetValues.end());

                std::cout << "\r" << completed << "/" << numEvaluations;
                if (addWarning)
                    std::cout << " Warning: " << testCase.id << "\n";
                else {
                    std::cout << std::flush;
                }
            }
        }
    };

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

    std::ofstream onsetDetectionValues(testUtils::getOutDir() / "onsetDetectionValues.py");
    testUtils::PrintPythonVector(onsetDetectionValues, onsetValues, "onsetValues");
    testUtils::PrintPythonVector(onsetDetectionValues, nonOnsetValues, "nonOnsetValues");
}

}  // namespace saint
