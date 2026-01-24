#pragma once

#include <filesystem>
#include <optional>
#include <vector>

namespace saint {
namespace testUtils {

struct Audio {
  const std::vector<float> data;
  const int sampleRate;
};

std::optional<Audio> fromWavFile(std::filesystem::path path);
bool toWavFile(std::filesystem::path path, const Audio &audio);

std::filesystem::path getEvalDir();
std::filesystem::path getOutDir();
} // namespace testUtils
} // namespace saint