#include "Utils.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

namespace saint {
std::string utils::getEnvironmentVariable(const char *var) {
#ifdef _WIN32
  char *buffer = nullptr;
  size_t size = 0;
  _dupenv_s(&buffer, &size, var);
  if (!buffer) {
    return "";
  } else {
    std::string result{buffer};
    free(buffer);
    return result;
  }
#else
  const char *value = std::getenv(var);
  return value ? std::string{value} : "";
#endif
}

bool utils::getEnvironmentVariableAsBool(const char *var) {
  auto str = getEnvironmentVariable(var);
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "1" || str == "true" || str == "on" || str == "yes" ||
         str == "y";
}

bool utils::isDebugBuild() {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif
}

float utils::getPitch(int noteNumber) {
  return 440 * std::pow(2.f, (noteNumber - 69) / 12.f);
}

float utils::getCrotchetsPerSample(float crotchetsPerSecond,
                                   int samplesPerSecond) {
  return (crotchetsPerSecond == 0 ? 120.f : crotchetsPerSecond) /
         static_cast<float>(samplesPerSecond);
}

std::vector<float> utils::getAnalysisWindow(int windowSize) {
  std::vector<float> window((size_t)windowSize);
  constexpr auto twoPi = 6.283185307179586f;
  const auto freq = twoPi / (float)windowSize;
  // TODO: make sure a rectangular window is tried.
  for (auto i = 0u; i < windowSize; ++i) {
    // A Hanning window.
    // For this use case and if there is not need for overlapping windows,
    // a flat-top might work as well.
    // window[i] = 1.f / fftSize;
    // i + 1 so that the tip of the window is at windowSize / 2, which is
    // convenient when taking the second half of it.
    window[i] = (1 - cosf((i + 1) * freq)) / 2;
  }
  return window;
}

} // namespace saint