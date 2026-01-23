#include "PitchDetectorImpl.h"

#include "PitchDetectorLogger.h"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace saint {

namespace fs = std::filesystem;

namespace {
void writeResultFile(const fs::path &outDir,
                     const std::filesystem::path &filenameStem,
                     const std::vector<std::optional<float>> &results) {
  std::ofstream resultFile(outDir / (filenameStem.string() + ".py"));
  resultFile << "results = [";
  auto separator = "";
  for (const auto &r : results) {
    if (r.has_value()) {
      resultFile << separator << *r;
    } else {
      resultFile << separator << "None";
    }
    separator = ",\n";
  }
  resultFile << "]\n";
}

void writeMarkedWavFile(const std::filesystem::path &filenameStem,
                        const testUtils::Audio &src, int markSample) {
  auto cpy = src.data;
  cpy[markSample] = 1.f;
  testUtils::toWavFile(testUtils::getOutDir() /
                           (filenameStem.string() + "_marked.wav"),
                       {cpy, src.sampleRate});
}
} // namespace

TEST(PitchDetectorImpl, testOnFiles) {
  const fs::path testFileDir = testUtils::getEvalDir() / "testFiles";
  std::vector<fs::path> testFiles;
  for (const auto &entry : fs::directory_iterator(testFileDir)) {
    if (entry.path().extension() == ".wav") {
      testFiles.push_back(entry.path());
    }
  }

  for (const auto &testFile : testFiles) {
    std::cout << "Processing " << testFile << "\n";
    const std::filesystem::path filenameStem = testFile.stem();
    constexpr auto blockSize = 512;
    const testUtils::Audio src = testUtils::fromWavFile(testFile);
    constexpr auto estimateIndex = 9;
    auto logger =
        std::make_unique<PitchDetectorLogger>(src.sampleRate, estimateIndex);
    const auto *loggerPtr = logger.get();
    PitchDetectorImpl sut(src.sampleRate, std::move(logger));
    std::vector<std::optional<float>> results;
    for (auto n = 0; n + blockSize < src.data.size(); n += blockSize) {
      std::vector<float> buffer(blockSize);
      std::vector<float *> channels(1);
      channels[0] = buffer.data();
      const auto result = sut.process(src.data.data() + n, blockSize);
      // If there is no update, we log 0.
      results.push_back(result);
    }

    writeResultFile(testUtils::getOutDir(), filenameStem, results);
    if (const auto index = loggerPtr->analysisAudioIndex())
      writeMarkedWavFile(filenameStem, src, *index);
  }
}
} // namespace saint
