#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <mutex>
#include <sstream>
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

// Clean-signal peak levels (dBFS) at which to test, parsed from the CLI knob
// `cleanPeaks=` (comma-separated). Defaults to {-10} (the historical fixed level).
// Varying the clean level exercises the onset detector's level-invariance: the
// adaptive threshold (flux > k*median) should detect onsets equally at any level,
// down to the level where onsetFluxAbsFloor starts to bind.
std::vector<float> parseCleanPeaks(const std::optional<std::string>& arg) {
    std::vector<float> peaks;
    if (arg.has_value()) {
        std::stringstream ss(*arg);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty())
                peaks.push_back(std::stof(token));
        }
    }
    if (peaks.empty())
        peaks.push_back(-10.f);
    return peaks;
}

// Prepare test cases with overlapping plucks: noisy = clean(peakDb) + noise +
// clean_delayed(peakDb). The clean signal is scaled to `peakDb` while the noise keeps
// its own RMS level (ambient noise is independent of how loud the instrument is), so
// lowering peakDb lowers the SNR exactly as a quiet/unplugged instrument would.
std::vector<TestCase> buildOnsetCasesAtLevel(const std::optional<std::string>& argTestCaseId,
                                             float peakDb) {
    auto testCases = prepareTestCases(argTestCaseId);

    std::vector<TestCase> out;
    out.reserve(testCases.size());
    for (auto& tc : testCases) {
        auto clean = testUtils::fromWavFile(tc.sample.file);
        if (!clean.has_value())
            continue;
        testUtils::scaleToPeak(clean->interleaved, peakDb);
        clean->interleaved.resize(tc.noisy.interleaved.size(), 0.f);

        const auto numChannels = tc.noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
        const int delaySamples = (tc.noisy.sampleRate * kDelaySeconds) * numChannels;

        // Rebuild the mix from the re-scaled clean so the noise level is unchanged.
        // Audio has a const sampleRate (not assignable), so replace only the samples;
        // sampleRate/channelFormat already match the note.
        tc.noisy.interleaved = clean->interleaved;
        testUtils::mixNoise(tc.noisy, tc.noise.data);
        for (size_t i = delaySamples; i < tc.noisy.interleaved.size(); ++i) {
            tc.noisy.interleaved[i] += clean->interleaved[i - delaySamples];
        }
        out.push_back(std::move(tc));
    }
    return out;
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

    const auto minFreq = getMinFreq(kTestTuning);

    const auto logFilePath = testUtils::getOutDir() / "onset_calibration.log";
    std::ofstream logFile(logFilePath);
    testUtils::TeeStream tee(std::cout, logFile);

    const auto argTestCaseId = getArgument<std::string>("testCaseId");
    const auto cleanPeaks = parseCleanPeaks(getArgument<std::string>("cleanPeaks"));
    // Onset-detector knobs (default to the production constants); overridable to sweep.
    const auto onsetK = getArgument<float>("onsetK").value_or(onsetFluxMedianMultiplier);
    const auto onsetAbsFloor = getArgument<float>("onsetAbsFloor").value_or(onsetFluxAbsFloor);

    // Cases that still fail separation with the spectral-flux onset strength: all
    // are home_2.wav (a loud household transient) at the loudest -40 dB level over
    // a quiet acoustic/classical-guitar pluck, where the transient genuinely beats
    // the weaker re-pluck. The previous broadband-energy detector blacklisted 30
    // cases; spectral flux fixes 24 of them (see onset-benchmark-results.md). The id
    // is identical across clean levels (only the clean scaling differs), so these
    // entries apply at every level.
    // clang-format off
    const std::vector<std::string> blacklist{
        "testFiles/notes/Admira_Classic/A2.wav | testFiles/noise/home_2.wav | -40",
        "testFiles/notes/Admira_Classic/D3.wav | testFiles/noise/home_2.wav | -40",
        "testFiles/notes/Admira_Classic/E2.wav | testFiles/noise/home_2.wav | -40",
        "testFiles/notes/Admira_Classic/E4.wav | testFiles/noise/home_2.wav | -40",
        "testFiles/notes/Admira_Classic/G3.wav | testFiles/noise/home_2.wav | -40",
        "testFiles/notes/Grand_Acoustic/iPhone_7/E2.wav | testFiles/noise/home_2.wav | -40",
    };
    // clang-format on

    // Reference-level (-10 dB) strength values, emitted for threshold derivation.
    std::vector<float> onsetValues;
    std::vector<float> nonOnsetValues;

    for (const auto peakDb : cleanPeaks) {
        const std::vector<TestCase> testCases = buildOnsetCasesAtLevel(argTestCaseId, peakDb);
        const auto numEvaluations = testCases.size();
        const bool isReferenceLevel = peakDb == -10.f;

        std::atomic<int> completedCount{0};
        std::mutex progressMutex;
        // Boolean isOnset behaviour (the detector's actual output): does an onset fire
        // within each labelled onset window, and how often outside them? These are the
        // level-invariance check - they should hold across clean levels until
        // onsetFluxAbsFloor starts to bind.
        std::atomic<int> onsetsDetected{0};
        std::atomic<int> onsetsTotal{0};
        std::atomic<int> falsePositives{0};
        std::atomic<int> warnings{0};

        tee << "[clean peak " << peakDb << " dB] evaluating " << numEvaluations
            << " test cases...\n";

        auto processTestCases = [&](size_t startIdx, size_t endIdx) {
            for (size_t idx = startIdx; idx < endIdx; ++idx) {
                const auto& testCase = testCases[idx];
                const auto& noisy = testCase.noisy;
                const auto blockSize = testCase.blockSize;

                OnsetDetector onsetDetector(noisy.sampleRate, noisy.channelFormat, blockSize,
                                            minFreq, OnsetDetectorConfig{onsetK, onsetAbsFloor});

                const auto numChannels = noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
                const auto numFrames = noisy.interleaved.size() / numChannels;

                std::vector<DebugOutput> debugOutputs;
                std::vector<char> isOnsets;  // the detector's actual boolean output per block

                auto filteredNoisy = noisy.interleaved;
                float* const noisyData = filteredNoisy.data();

                for (size_t i = 0; i + blockSize < numFrames; i += blockSize) {
                    DebugOutput debugOutput;
                    const bool fired = onsetDetector.process(noisyData + i * numChannels, &debugOutput);
                    isOnsets.push_back(fired ? 1 : 0);
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

                // The detector decides on the level-invariant ratio flux/median (an onset
                // is ratio > k), so the separation metric and the derived threshold
                // operate on the ratio, not the raw (level-dependent) flux.
                const auto onsetStrengths = extractDebugOutput(debugOutputs, "onsetRatio");
                const auto a = onsetStrengths.begin() +
                               std::max<int>(firstOnsetBlockIndex - marginBeforeBlocks, 0);
                const auto b = onsetStrengths.begin() + firstOnsetBlockIndex + marginAfterBlocks;
                const auto c = onsetStrengths.begin() + secondOnsetBlockIndex - marginBeforeBlocks;
                const auto d =
                    onsetStrengths.begin() +
                    std::min<int>(secondOnsetBlockIndex + marginAfterBlocks, onsetStrengths.size());

                const auto firstMax = *std::max_element(a, b);
                const auto secondMax = *std::max_element(c, d);

                const auto inBlacklist =
                    std::find(blacklist.begin(), blacklist.end(), testCase.id) != blacklist.end();

                // Boolean isOnset detection within each onset window, and false positives
                // outside both windows.
                const int w1lo = std::max(firstOnsetBlockIndex - marginBeforeBlocks, 0);
                const int w1hi = firstOnsetBlockIndex + marginAfterBlocks;
                const int w2lo = secondOnsetBlockIndex - marginBeforeBlocks;
                const int w2hi = secondOnsetBlockIndex + marginAfterBlocks;
                const auto firedIn = [&](int lo, int hi) {
                    lo = std::max(lo, 0);
                    hi = std::min<int>(hi, static_cast<int>(isOnsets.size()));
                    for (int k = lo; k < hi; ++k)
                        if (isOnsets[k])
                            return true;
                    return false;
                };
                onsetsTotal += 2;
                onsetsDetected += (firedIn(w1lo, w1hi) ? 1 : 0) + (firedIn(w2lo, w2hi) ? 1 : 0);
                int caseFalsePositives = 0;
                for (int k = 0; k < static_cast<int>(isOnsets.size()); ++k) {
                    if (!isOnsets[k])
                        continue;
                    const bool inW1 = k >= w1lo && k < w1hi;
                    const bool inW2 = k >= w2lo && k < w2hi;
                    if (!inW1 && !inW2)
                        ++caseFalsePositives;
                }
                if (!inBlacklist)
                    falsePositives += caseFalsePositives;

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

                    std::vector<float> newNonOnsetValues;
                    newNonOnsetValues.insert(newNonOnsetValues.end(), onsetStrengths.begin(), a);
                    newNonOnsetValues.insert(newNonOnsetValues.end(), b, c);
                    newNonOnsetValues.insert(newNonOnsetValues.end(), d, onsetStrengths.end());

                    const auto leastOfMax = std::min(firstMax, secondMax);
                    const auto addWarning =
                        !inBlacklist &&
                        std::any_of(newNonOnsetValues.begin(), newNonOnsetValues.end(),
                                    [leastOfMax](float v) { return v > leastOfMax; });
                    if (addWarning)
                        ++warnings;

                    // Only the reference level seeds the threshold-derivation vectors.
                    if (isReferenceLevel) {
                        onsetValues.push_back(firstMax);
                        onsetValues.push_back(secondMax);
                        nonOnsetValues.insert(nonOnsetValues.end(), newNonOnsetValues.begin(),
                                              newNonOnsetValues.end());
                    }

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

        const auto detPct = onsetsTotal > 0 ? 100.0 * onsetsDetected / onsetsTotal : 0.0;
        tee << "\n[clean peak " << peakDb << " dB] onsets detected " << onsetsDetected << "/"
            << onsetsTotal << " (" << detPct << "%), false-positive firings " << falsePositives
            << ", ratio-separation warnings " << warnings << "\n";
    }

    std::ofstream onsetDetectionValues(testUtils::getOutDir() / "onsetDetectionValues.py");
    testUtils::PrintPythonVector(onsetDetectionValues, onsetValues, "onsetValues");
    testUtils::PrintPythonVector(onsetDetectionValues, nonOnsetValues, "nonOnsetValues");
}

}  // namespace saint
