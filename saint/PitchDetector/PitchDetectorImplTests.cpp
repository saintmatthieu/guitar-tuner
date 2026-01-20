#include "FormantShifterLogger.h"
#include "PitchDetectorImpl.h"

#include "testUtils.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace saint {

namespace fs = std::filesystem;

TEST(PitchDetectorImpl, testOnFiles) {
  const fs::path testFileDir = testUtils::getRootDir() + "testFiles/";
  std::vector<fs::path> testFiles;
  for (const auto &entry : fs::directory_iterator(testFileDir)) {
    if (entry.path().extension() == ".wav") {
      testFiles.push_back(entry.path());
    }
  }

  for (const auto &testFile : testFiles) {
    std::cout << "Processing " << testFile << "\n";
    const auto filenameStem = testFile.stem().string();
    constexpr auto blockSize = 512;
    const testUtils::Audio src = testUtils::fromWavFile(testFile);
    constexpr auto logTimeInSeconds = 1.876;
    auto logger = std::make_unique<FormantShifterLogger>(src.sampleRate, 82432);
    PitchDetectorImpl sut(src.sampleRate, std::move(logger));
    std::vector<float> results;
    for (auto n = 0; n + blockSize < src.data.size(); n += blockSize) {
      std::vector<float> buffer(blockSize);
      std::vector<float *> channels(1);
      channels[0] = buffer.data();
      const auto result = sut.process(src.data.data() + n, blockSize);
      // If there is no update, we log 0.
      results.push_back(result.has_value() ? *result : 0.f);
    }

    std::ofstream resultFile(testUtils::getOutDir() + filenameStem + ".py");
    resultFile << "results = [";
    auto separator = "";
    for (const auto r : results) {
      resultFile << separator << r;
      separator = ",\n";
    }
    resultFile << "]\n";
    // break; // Only one file while debugging
  }
}

} // namespace saint
