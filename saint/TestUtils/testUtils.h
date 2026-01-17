#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace saint {
namespace testUtils {

struct Audio {
  const std::vector<float> data;
  const int sampleRate;
};

Audio fromWavFile(std::filesystem::path path);

std::string getInputFilePath();
std::string getRootDir();
std::string getOutDir();
} // namespace testUtils
} // namespace saint