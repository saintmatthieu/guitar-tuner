// Data collection for the autocorrelation peak census.
//
// Runs only the front half of the pipeline (Preprocessor -> FrequencyDomainTransformer ->
// AutocorrPitchDetector), unconstrained, over the same test-case matrix as the benchmark,
// and dumps every local maximum of the autocorrelation function per block. The resulting
// CSVs (eval/out/peakCensus_*.csv) feed eval/peakCensus.py, which computes statistics of
// the peak landscape relative to the ground-truth lag (octave/fifth confusers, recall of
// the true peak, etc.).
//
// Output files:
//   peakCensus_cases.csv   one row per test case (note x noise x SNR)
//   peakCensus_blocks.csv  one row per processed block: what the current max-picking
//                          returns (unconstrained) and the block's onset-relative time
//   peakCensus_peaks.csv   one row per autocorrelation local maximum
//
// Optional arguments (same convention as the benchmark):
//   testCaseId=<id>   collect a single test case only

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

#include "AutocorrPitchDetector.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetectorLoggerInterface.h"
#include "PitchDetectorUtils.h"
#include "Preprocessor.h"
#include "TestCaseUtils.h"
#include "testUtils.h"

namespace saint {
namespace {
// Captures the buffers AutocorrPitchDetector logs on each process() call.
class XcorrCapturingLogger : public PitchDetectorLoggerInterface {
   public:
    void SamplesRead(int) override {}
    bool StartNewEstimate() override {
        return false;
    }
    void Log(int, const char*) const override {}
    void Log(float, const char*) const override {}
    void Log(const float* samples, size_t size, const char* name,
             const std::function<float(float)>&) const override {
        if (std::strcmp(name, "xcorr") == 0) {
            xcorr.assign(samples, samples + size);
        } else if (windowXcorr.empty() && std::strcmp(name, "windowXcorr") == 0) {
            windowXcorr.assign(samples, samples + size);
        }
    }
    void Log(const std::complex<float>*, size_t, const char*,
             const std::function<float(const std::complex<float>&)>&) const override {}
    void EndNewEstimate(std::complex<float>*, size_t) override {}

    mutable std::vector<float> xcorr;
    mutable std::vector<float> windowXcorr;
};

struct Peak {
    float lag = 0.f;        // quad-refined, in samples
    float raw = 0.f;        // r(lag) / r(0), refined
    float corrected = 0.f;  // raw / windowXcorr[index] (the presence-score normalization)
    bool preNegativeCrossing = false;
};

// Extracts all local maxima of the autocorrelation up to lastSearchIndex, mirroring the
// conventions of AutocorrPitchDetector::process: peaks before the first negative crossing
// (which the detector's search skips) are kept but flagged.
std::vector<Peak> extractPeaks(const std::vector<float>& xcorr,
                               const std::vector<float>& windowXcorr, int lastSearchIndex) {
    std::vector<Peak> peaks;
    bool wentNegative = false;
    for (int i = 1; i + 1 < lastSearchIndex; ++i) {
        wentNegative |= xcorr[i] < 0;
        if (xcorr[i] <= 0.f || xcorr[i] <= xcorr[i - 1] || xcorr[i] <= xcorr[i + 1]) {
            continue;
        }
        const auto y0 = xcorr[i - 1];
        const auto y1 = xcorr[i];
        const auto y2 = xcorr[i + 1];
        const auto delta = utils::quadFit(&xcorr[i - 1]);
        const auto a = 0.5f * (y0 - 2 * y1 + y2);
        const auto b = 0.5f * (y2 - y0);
        const auto refinedValue = y1 + b * delta + a * delta * delta;
        Peak peak;
        peak.lag = i + delta;
        peak.raw = refinedValue;
        peak.corrected = windowXcorr[i] > 0.f ? refinedValue / windowXcorr[i] : 0.f;
        peak.preNegativeCrossing = !wentNegative;
        peaks.push_back(peak);
    }
    return peaks;
}

std::string formatFloat(const char* fmt, double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, value);
    return buffer;
}
}  // namespace

TEST(AutocorrPeakCensus, collect) {
    const auto argTestCaseId = getArgument<std::string>("testCaseId");
    const std::vector<TestCase> testCases = prepareTestCases(argTestCaseId);
    ASSERT_FALSE(testCases.empty());

    const auto outDir = testUtils::getOutDir();
    std::ofstream casesFile(outDir / "peakCensus_cases.csv");
    std::ofstream blocksFile(outDir / "peakCensus_blocks.csv");
    std::ofstream peaksFile(outDir / "peakCensus_peaks.csv");
    ASSERT_TRUE(casesFile.is_open() && blocksFile.is_open() && peaksFile.is_open());

    casesFile << "caseIdx,noteFile,noiseFile,snrDb,trueFreq,truthStart,truthEnd,sampleRate,"
                 "blockSize,fftSize,windowSize\n";
    blocksFile << "caseIdx,blockIdx,tOnset,algoFreq,presenceScore\n";
    peaksFile << "caseIdx,blockIdx,lag,raw,corr,preNeg\n";

    std::mutex writeMutex;
    std::atomic<int> completedCount{0};
    const auto numEvaluations = testCases.size();

    std::cout << "Collecting peak census over " << numEvaluations << " test cases..."
              << std::endl;

    auto processTestCases = [&](size_t startIdx, size_t endIdx) {
        for (size_t idx = startIdx; idx < endIdx; ++idx) {
            const auto& testCase = testCases[idx];
            const auto& sample = testCase.sample;
            const auto& noisy = testCase.noisy;
            const auto blockSize = testCase.blockSize;
            const auto sampleRate = noisy.sampleRate;

            XcorrCapturingLogger logger;
            const auto minFreq = getMinFreq(kTestTuning);
            Preprocessor preprocessor(sampleRate, noisy.channelFormat, blockSize);
            FrequencyDomainTransformer transformer(sampleRate, noisy.channelFormat, blockSize,
                                                   minFreq, logger);
            AutocorrPitchDetector detector(sampleRate, transformer.fftSize(),
                                           transformer.window(), minFreq, logger);
            // Same bound as AutocorrPitchDetector's unconstrained search.
            const auto lastSearchIndex = std::min(
                transformer.fftSize() / 2, static_cast<int>(sampleRate / minFreq));

            const auto numChannels = noisy.channelFormat == ChannelFormat::Mono ? 1 : 2;
            const auto numFrames = noisy.interleaved.size() / numChannels;
            const auto* noisyData = noisy.interleaved.data();

            std::string blocksRows;
            std::string peaksRows;
            const auto casePrefix = std::to_string(idx) + ",";

            auto blockIdx = 0;
            for (auto i = 0u; i + blockSize < numFrames; i += blockSize, ++blockIdx) {
                const auto processedAudio =
                    preprocessor.processBlock(noisyData + i * numChannels);
                const std::vector<std::complex<float>> freq =
                    transformer.process(processedAudio.data());

                auto presenceScore = 0.f;
                const auto algoFreq = detector.process(freq, &presenceScore);

                // Timestamp of the analysis-window centre, relative to the note attack.
                const auto currentTime =
                    static_cast<double>(static_cast<int>(i) + blockSize -
                                        transformer.delaySamples()) /
                    sampleRate;
                const auto tOnset = currentTime - sample.truth.startTime;

                const auto blockPrefix = casePrefix + std::to_string(blockIdx) + ",";
                blocksRows += blockPrefix + formatFloat("%.3f", tOnset) + "," +
                              formatFloat("%.3f", algoFreq) + "," +
                              formatFloat("%.4g", presenceScore) + "\n";

                const std::vector<Peak> peaks =
                    extractPeaks(logger.xcorr, logger.windowXcorr, lastSearchIndex);
                for (const auto& peak : peaks) {
                    peaksRows += blockPrefix + formatFloat("%.2f", peak.lag) + "," +
                                 formatFloat("%.4g", peak.raw) + "," +
                                 formatFloat("%.4g", peak.corrected) + "," +
                                 (peak.preNegativeCrossing ? "1" : "0") + "\n";
                }
            }

            const auto evalDir = testUtils::getEvalDir();
            std::stringstream caseRow;
            caseRow << idx << "," << fs::relative(sample.file, evalDir).string() << ","
                    << fs::relative(testCase.noise.file, evalDir).string() << ","
                    << testCase.noise.rmsDb << "," << sample.truth.frequency << ","
                    << sample.truth.startTime << "," << sample.truth.endTime << ","
                    << sampleRate << "," << blockSize << "," << transformer.fftSize() << ","
                    << transformer.windowSizeSamples() << "\n";

            {
                std::lock_guard<std::mutex> lock(writeMutex);
                casesFile << caseRow.str();
                blocksFile << blocksRows;
                peaksFile << peaksRows;
                std::cout << "\r" << ++completedCount << "/" << numEvaluations << std::flush;
            }
        }
    };

    const auto numThreads = std::max(1u, std::thread::hardware_concurrency());
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
    std::cout << "\nDone. Output in " << outDir.string() << "/peakCensus_*.csv" << std::endl;
}
}  // namespace saint
