#pragma once

#include <string>

namespace saint {
namespace utils {
std::string getEnvironmentVariable(const char *);
bool getEnvironmentVariableAsBool(const char *);
bool isDebugBuild();
float getPitch(int noteNumber);
float getCrotchetsPerSample(float crotchetsPerSecond, int samplesPerSecond);
} // namespace utils
} // namespace saint
