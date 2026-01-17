#include "Utils.h"
#include <algorithm>
#include <cctype>
#include <cmath>

namespace saint {
namespace utils {
std::string getEnvironmentVariable(const char *var) {
  char *buffer = nullptr;
  size_t size = 0;
  _dupenv_s(&buffer, &size, var);
  if (!buffer) {
    return "";
  } else {
    return std::string{buffer};
  }
}

bool getEnvironmentVariableAsBool(const char *var) {
  auto str = getEnvironmentVariable(var);
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "1" || str == "true" || str == "on" || str == "yes" ||
         str == "y";
}

bool isDebugBuild() {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif
}

float getPitch(int noteNumber) {
  return 440 * std::powf(2, (noteNumber - 69) / 12.f);
}

float getCrotchetsPerSample(float crotchetsPerSecond, int samplesPerSecond) {
  return (crotchetsPerSecond == 0 ? 120.f : crotchetsPerSecond) /
         static_cast<float>(samplesPerSecond);
}

} // namespace utils
} // namespace saint