#pragma once

#include <string>
#include <vector>

namespace saint {
namespace utils {
std::string getEnvironmentVariable(const char *);
bool getEnvironmentVariableAsBool(const char *);
bool isDebugBuild();
float getPitch(int noteNumber);
float getCrotchetsPerSample(float crotchetsPerSecond, int samplesPerSecond);
std::vector<float> getAnalysisWindow(int windowSize);
} // namespace utils
} // namespace saint
