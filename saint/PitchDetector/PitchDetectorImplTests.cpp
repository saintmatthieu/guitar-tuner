#include "PitchDetectorImpl.h"

#include "PitchDetectorLogger.h"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace saint {

namespace fs = std::filesystem;

namespace {
struct Truth {
  const double startTime;
  const double endTime;
  const float frequency;
};

struct Sample {
  const fs::path file;
  const Truth truth;
};

enum class ActivityResult {
  TruePositive,
  TrueNegative,
  FalsePositive,
  FalseNegative,
};

struct ActivityRates {
  double tp = 0;
  double tn = 0;
  double fp = 0;
  double fn = 0;
};

struct DetectionResult {
  ActivityResult activity = static_cast<ActivityResult>(0);
  std::optional<float> frequency;
};

struct RocResult {
  RocResult(bool t, double s) : truth(t), score(s) {}
  bool truth = false;
  double score = 0.0;
};

float getTrueFrequency(const std::filesystem::path &filePath) {
  const auto filename = filePath.stem().string();
  // File name in the form of <note name><note octave>:
  const auto noteName = filename.substr(0, filename.size() - 1);
  const auto noteOctave = filename.back() - '0';
  const std::vector<std::string> noteNames{
      "C", "D", "E", "F", "G", "A", "B",
  };
  const std::vector<int> noteOffsets{
      0, 2, 4, 5, 7, 9, 11,
  };
  const auto it = std::find_if(
      noteNames.begin(), noteNames.end(), [&noteName](const std::string &name) {
        return std::equal(
            name.begin(), name.end(), noteName.begin(), noteName.end(),
            [](char a, char b) { return std::tolower(a) == std::tolower(b); });
      });

  if (it == noteNames.end()) {
    return 0.f;
  }

  const int noteOffset = noteOffsets[std::distance(noteNames.begin(), it)];
  const int midiNote = (noteOctave + 1) * 12 + noteOffset;
  return 440.f * std::pow(2.f, (midiNote - 69) / 12.f);
}

fs::path getFileShortName(const fs::path &filePath) {
  return filePath.parent_path().stem() / filePath.stem();
}

void writeResultFile(const Sample &sample,
                     const std::vector<DetectionResult> &results,
                     float &rmsErrorCents) {

  const auto filenameStem = getFileShortName(sample.file);
  const auto filename =
      testUtils::getOutDir() / (filenameStem.string() + "_results.py");

  if (!fs::exists(filename.parent_path())) {
    fs::create_directories(filename.parent_path());
  }

  std::ofstream resultFile(filename);

  rmsErrorCents = 0.f;
  std::vector<float> errorCents;
  for (const auto &r : results) {
    if (r.frequency.has_value() && *r.frequency > 0.f) {
      const auto e =
          1200.f * std::log2((*r.frequency) / sample.truth.frequency);
      rmsErrorCents += e * e;
      errorCents.push_back(e);
    }
  }
  if (!errorCents.empty()) {
    rmsErrorCents =
        std::sqrt(rmsErrorCents / static_cast<float>(errorCents.size()));
  }
  resultFile << "rmsErrorCents = " << rmsErrorCents << "\n";

  resultFile << "results = [";
  auto separator = "";
  for (const auto &e : errorCents) {
    resultFile << separator << e;
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

void scaleToRms(std::vector<float> &data, float targetRmsDb) {
  float sumSquares = 0.f;
  for (const auto sample : data) {
    sumSquares += sample * sample;
  }
  const float currentRms =
      std::sqrt(sumSquares / static_cast<float>(data.size()));
  const float targetRms = std::pow(10.f, targetRmsDb / 20.f);
  const float scale = targetRms / currentRms;
  for (auto &sample : data) {
    sample *= scale;
  }
}

std::optional<Sample> getSampleFromFile(const fs::path &filePath) {
  std::cout << "Reading sample from " << filePath << "\n";

  const auto trueFreq = getTrueFrequency(filePath);
  if (trueFreq == 0.f) {
    std::cerr << "Could not determine true frequency for " << filePath << "\n";
    return std::nullopt;
  }

  const auto labelPath =
      filePath.parent_path() / (filePath.stem().string() + ".txt");
  if (!fs::exists(labelPath)) {
    std::cerr << "Could not find label file: " << labelPath << "\n";
    return std::nullopt;
  }

  // Formatted in Audacity label format: <start time>tab<end time> (ignore
  // possible label text)
  std::ifstream labelFile(labelPath);
  std::string line;
  if (!std::getline(labelFile, line)) {
    std::cerr << "Could not read label file: " << labelPath << "\n";
    return std::nullopt;
  }

  const auto tabPos = line.find('\t');
  if (tabPos == std::string::npos) {
    std::cerr << "Could not parse label file: " << labelPath << "\n";
    return std::nullopt;
  }
  const auto startTime = std::stof(line.substr(0, tabPos));
  const auto endTime = std::stof(line.substr(tabPos + 1));

  return Sample{filePath, Truth{startTime, endTime, trueFreq}};
}

constexpr auto precision = std::numeric_limits<double>::digits10 + 1;

template <typename T>
bool ValueIsUnchanged(const std::filesystem::path &filePath, T previousValue,
                      T newValue, T tolerance = 0) {
  std::cout << "Checking value in " << filePath << "\n";
  assert(std::filesystem::exists(filePath));
  const auto hasChanged = std::abs(newValue - previousValue) > tolerance;
  if (hasChanged) {
    std::ofstream file{filePath};
    file << std::setprecision(precision) << newValue;
  }
  return !hasChanged;
}
} // namespace

TEST(PitchDetectorImpl, benchmarking) {

  std::vector<Sample> samples;
  // Check if a specific test file is provided as argument
  const auto &argv = ::testing::internal::GetArgvs();
  for (size_t i = 1; i < argv.size(); ++i) {
    const fs::path testFile(argv[i]);
    if (!fs::exists(testFile) || testFile.extension() != ".wav") {
      continue;
    }
    auto sample = getSampleFromFile(testFile);
    if (sample.has_value()) {
      samples.push_back(std::move(*sample));
    }
  }

  // If no specific file was provided, scan the test directory
  if (samples.empty()) {
    const fs::path testFileDir =
        testUtils::getEvalDir() / "testFiles" / "notes";
    for (const auto &entry : fs::recursive_directory_iterator(testFileDir)) {
      if (entry.path().extension() != ".wav") {
        continue;
      }
      auto sample = getSampleFromFile(entry.path());
      if (sample.has_value()) {
        samples.push_back(std::move(*sample));
      }
    }
  }

  std::vector<fs::path> noiseFiles;
  for (const auto &entry : fs::recursive_directory_iterator(
           testUtils::getEvalDir() / "testFiles" / "noise")) {
    if (entry.path().extension() == ".wav") {
      noiseFiles.push_back(entry.path());
    }
  }

  struct Noise {
    const fs::path file;
    const char *rmsDb;
    std::vector<float> data;
  };

  const std::vector<const char *> noiseRmsDb{"-40", "-30"};
  std::vector<Noise> noiseData;
  for (const auto &noiseFile : noiseFiles) {
    auto noiseAudio = testUtils::fromWavFile(noiseFile);
    if (!noiseAudio.has_value()) {
      continue;
    }
    for (const auto rmsDb : noiseRmsDb) {
      const float dB = std::stof(rmsDb);
      scaleToRms(noiseAudio->data, dB);
      noiseData.push_back(Noise{noiseFile, rmsDb, noiseAudio->data});
    }
  }

  std::vector<float> rmsErrors;
  std::vector<RocResult> rocResults;

  for (const auto &sample : samples) {
    const auto &testFile = sample.file;
    std::cout << "Processing " << testFile << "\n";

    const std::filesystem::path cleanFile =
        testFile.parent_path().stem() / testFile.stem();
    constexpr auto blockSize = 512;
    const std::optional<testUtils::Audio> clean =
        testUtils::fromWavFile(testFile);

    if (!clean.has_value()) {
      std::cerr << "Could not read file: " << testFile << "\n";
      continue;
    }

    for (const auto &noise : noiseData) {
      auto noisy = clean->data;
      // Mix noise into clean signal
      const auto noiseSize = noise.data.size();
      auto n = 0;
      for (auto &sample : noisy) {
        sample += noise.data[n];
        n = (n + 1) % noiseSize;
      }

      constexpr auto estimateIndex = 721;
      auto logger = std::make_unique<PitchDetectorLogger>(clean->sampleRate,
                                                          estimateIndex);
      const auto *loggerPtr = logger.get();
      PitchDetectorImpl sut(clean->sampleRate, std::move(logger));
      std::vector<DetectionResult> results;
      std::optional<float> currentEstimate;
      for (auto n = 0; n + blockSize < noisy.size(); n += blockSize) {
        std::vector<float> buffer(blockSize);
        std::vector<float *> channels(1);
        channels[0] = buffer.data();
        auto presenceScore = 0.f;
        auto result = sut.process(noisy.data() + n, blockSize, &presenceScore);
        const auto currentTime =
            static_cast<double>(n + blockSize) / clean->sampleRate;
        const auto actual =
            (result.has_value() && *result > 0) ||
            (currentEstimate.has_value() && *currentEstimate > 0);
        const auto expected = (currentTime >= sample.truth.startTime) &&
                              (currentTime <= sample.truth.endTime);
        ActivityResult activity;
        if (actual && expected) {
          activity = ActivityResult::TruePositive;
        } else if (!actual && !expected) {
          activity = ActivityResult::TrueNegative;
        } else if (actual && !expected) {
          activity = ActivityResult::FalsePositive;
        } else {
          activity = ActivityResult::FalseNegative;
        }
        results.push_back({activity, result});
        if (result.has_value()) {
          rocResults.emplace_back(expected, presenceScore);
        }
      }

      const auto filename = cleanFile.string() + "_with_" +
                            noise.file.stem().string() + "_at_" + noise.rmsDb +
                            "dB";
      const auto outWavName =
          testUtils::getOutDir() / "wav" / (filename + ".wav");
      testUtils::toWavFile(outWavName, {noisy, clean->sampleRate});

      rmsErrors.push_back(0.f);
      writeResultFile(sample, results, rmsErrors.back());

      std::cout << "RMS error for " << getFileShortName(testFile)
                << " with noise " << getFileShortName(noise.file) << " at "
                << noise.rmsDb << " dB: " << rmsErrors.back() << " cents\n";

      if (const auto index = loggerPtr->analysisAudioIndex())
        writeMarkedWavFile(filename, *clean, *index);
    }
  }

  auto rmsAvg = 0.f;
  for (const auto e : rmsErrors) {
    rmsAvg += e;
  }
  if (!rmsErrors.empty()) {
    rmsAvg /= static_cast<float>(rmsErrors.size());
  }
  std::cout << "Average RMS error across all tests: " << rmsAvg << " cents\n";

  constexpr auto allowedFalsePositiveRate = 0.05;
  const testUtils::RocInfo rocInfo =
      testUtils::GetRocInfo<RocResult>(rocResults, allowedFalsePositiveRate);

  std::ofstream rocFile(testUtils::getOutDir() / "roc_curve.py");
  rocFile << "AUC = " << rocInfo.areaUnderCurve << "\n";
  rocFile << "threshold = " << rocInfo.threshold << "\n";
  rocFile << "allowedFalsePositiveRate = " << allowedFalsePositiveRate << "\n";
  testUtils::PrintPythonVector(rocFile, rocInfo.falsePositiveRates,
                               "falsePositiveRates");
  testUtils::PrintPythonVector(rocFile, rocInfo.truePositiveRates,
                               "truePositiveRates");

  constexpr auto previousAuc = 0.84327520718232096;
  constexpr auto comparisonTolerance = 0.01;
  const auto classifierQualityIsUnchanged = ValueIsUnchanged(
      testUtils::getEvalDir() / "BenchmarkingOutput" / "AUC.txt", previousAuc,
      rocInfo.areaUnderCurve, comparisonTolerance);

  // If it changes and it's for the better, then it's probably a good thing, but
  // let's keep an eye on it anyway. If it's for the worse, then either there is
  // a good reason or we have a regression.
  EXPECT_TRUE(classifierQualityIsUnchanged)
      << "Classifier quality has changed! Previous AUC: " << previousAuc
      << ", new AUC: " << rocInfo.areaUnderCurve;
}
} // namespace saint
