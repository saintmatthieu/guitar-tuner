#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>

namespace saint {
std::string utils::getEnvironmentVariable(const char* var) {
#ifdef _WIN32
    char* buffer = nullptr;
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
    const char* value = std::getenv(var);
    return value ? std::string{value} : "";
#endif
}

bool utils::getEnvironmentVariableAsBool(const char* var) {
    auto str = getEnvironmentVariable(var);
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return str == "1" || str == "true" || str == "on" || str == "yes" || str == "y";
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

float utils::getCrotchetsPerSample(float crotchetsPerSecond, int samplesPerSecond) {
    return (crotchetsPerSecond == 0 ? 120.f : crotchetsPerSecond) /
           static_cast<float>(samplesPerSecond);
}

namespace {
std::pair<float, float> getCoefs(utils::WindowType type) {
    switch (type) {
        case utils::WindowType::Hann:
            return {1.f, 1.f};
        case utils::WindowType::Hamming:
            // Found in PhD thesis Matthieu Hodgkinson @NUIM
            // https://mural.maynoothuniversity.ie/id/eprint/3910/1/thesis.pdf
            return {1.f, 349.f / 407.f};
        default:
            assert(false);
            return getCoefs(utils::WindowType::Hann);
    }
}
}  // namespace

std::vector<float> utils::getAnalysisWindow(int windowSize, WindowType type) {
    std::vector<float> window((size_t)windowSize);
    constexpr auto twoPi = 6.283185307179586f;
    const auto freq = twoPi / (float)windowSize;

    const auto [a0, a1] = getCoefs(type);
    auto sum = 0.f;
    for (auto i = 0u; i < windowSize; ++i) {
        // i + 1 so that the tip of the window is at windowSize / 2, which is
        // convenient when taking the second half of it.
        window[i] = a0 - a1 * cosf((i + 1) * freq);
        sum += window[i];
    }
    std::transform(window.begin(), window.end(), window.begin(),
                   [sum](float x) { return x / sum; });
    return window;
}

}  // namespace saint