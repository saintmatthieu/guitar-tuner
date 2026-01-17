#include "DummyFormantShifterLogger.h"
#include "FormantShifterLogger.h"
#include "PitchDetectorImpl.h"

#include "pffft.hpp"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace saint {

namespace fs = std::filesystem;

namespace {
std::vector<float> makeNyquistWave(int numSamples) {
  std::vector<float> nyquist(numSamples);
  for (auto i = 0; i < numSamples; ++i) {
    nyquist[i] = i % 2 == 0 ? 1 : -1;
  }
  return nyquist;
}
} // namespace

// TEST(PitchDetectorImpl, firstPfftBinIsDcAndNyquist) {
//   PitchDetectorImpl sut(44100, std::nullopt, std::nullopt,
//   std::make_unique<DummyFormantShifterLogger>()); constexpr auto blockSize =
//   512; const auto audio = makeNyquistWave(blockSize); pffft::Fft<float>
//   fftEngine(blockSize); std::vector<pffft::Fft<float>::Complex> fft(blockSize
//   / 2); fftEngine.forward(audio.data(), fft.data());
//   EXPECT_FLOAT_EQ(fft.data()[0].real(), 0);
//   EXPECT_FLOAT_EQ(fft.data()[0].imag(), blockSize);
// }

TEST(PitchDetectorImpl, stuff) {
  const fs::path testFileDir = testUtils::getRootDir() + "testFiles/";
  std::vector<fs::path> testFiles;
  // found all wav files in testFileDir
  for (const auto &entry : fs::directory_iterator(testFileDir)) {
    if (entry.path().extension() == ".wav") {
      testFiles.push_back(entry.path());
    }
  }

  for (const auto &testFile : testFiles) {
    // const auto debugCb = testUtils::getPitchDetectorDebugCb();
    const auto filenameStem = testFile.stem().string();
    constexpr auto blockSize = 512;
    constexpr auto sampleRate = 44100;
    const testUtils::Audio src = testUtils::fromWavFile(testFile);
    assert(src.sampleRate ==
           sampleRate); // some 44.1kHz assumptions are
                        // currently in the implementation ...
    constexpr auto logTimeInSeconds = 2.474;
    auto logger = std::make_unique<FormantShifterLogger>(
        sampleRate, logTimeInSeconds * sampleRate);
    constexpr auto A1Frequency = 55.0f;
    PitchDetectorImpl sut(sampleRate, A1Frequency, std::move(logger));
    std::ofstream resultFile(testUtils::getOutDir() + filenameStem + ".txt");
    for (auto n = 0; n + blockSize < src.data.size(); n += blockSize) {
      std::vector<float> buffer(blockSize);
      std::vector<float *> channels(1);
      channels[0] = buffer.data();
      // std::fill(buffer.begin(), buffer.end(), 1.f);
      const auto result = sut.process(src.data.data() + n, blockSize);
      resultFile << (result.has_value() ? *result : 0.f) << "\n";
    }
  }
}

} // namespace saint
