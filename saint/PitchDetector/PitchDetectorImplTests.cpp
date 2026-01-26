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
  const auto filename = outDir / (filenameStem.string() + ".py");
  // If outDir does not exist, create it.
  if (!fs::exists(filename.parent_path())) {
    fs::create_directories(filename.parent_path());
  }
  std::ofstream resultFile(filename);
  resultFile << "results = [";
  auto separator = "";
  for (const auto &r : results) {
    constexpr auto logNoneAsZero = false;
    if (r.has_value()) {
      resultFile << separator << *r;
      separator = ",\n";
    } else if (logNoneAsZero) {
      resultFile << separator << "0";
      separator = ",\n";
    }
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

  // Check if a specific test file is provided as argument
  std::vector<fs::path> testFiles;
  const auto &argv = ::testing::internal::GetArgvs();
  for (size_t i = 1; i < argv.size(); ++i) {
    fs::path testFile(argv[i]);
    if (fs::exists(testFile) && testFile.extension() == ".wav") {
      testFiles.push_back(testFile);
    }
  }

  // If no specific file was provided, scan the test directory
  if (testFiles.empty()) {
    const fs::path testFileDir =
        testUtils::getEvalDir() / "testFiles" / "notes";
    for (const auto &entry : fs::recursive_directory_iterator(testFileDir)) {
      if (entry.path().extension() == ".wav") {
        testFiles.push_back(entry.path());
      }
    }
  }

  for (const auto &testFile : testFiles) {
    std::cout << "Processing " << testFile << "\n";
    const std::filesystem::path filename =
        testFile.parent_path().stem() / testFile.stem();
    constexpr auto blockSize = 512;
    const std::optional<testUtils::Audio> src =
        testUtils::fromWavFile(testFile);
    if (!src.has_value()) {
      std::cerr << "Could not read file: " << testFile << "\n";
      continue;
    }
    constexpr auto estimateIndex = 6;
    auto logger =
        std::make_unique<PitchDetectorLogger>(src->sampleRate, estimateIndex);
    const auto *loggerPtr = logger.get();
    PitchDetectorImpl sut(src->sampleRate, std::move(logger));
    std::vector<std::optional<float>> results;
    for (auto n = 0; n + blockSize < src->data.size(); n += blockSize) {
      std::vector<float> buffer(blockSize);
      std::vector<float *> channels(1);
      channels[0] = buffer.data();
      const auto result = sut.process(src->data.data() + n, blockSize);
      results.push_back(result);
    }

    writeResultFile(testUtils::getOutDir(), filename, results);
    if (const auto index = loggerPtr->analysisAudioIndex())
      writeMarkedWavFile(filename, *src, *index);
  }
}
} // namespace saint
