#pragma once

#include <array>
#include <string>
#include <vector>

namespace saint {
namespace utils {
enum class WindowType { Hann, Hamming, MinimumThreeTerm, _count };
constexpr auto numWindowTypes = static_cast<size_t>(WindowType::_count);
constexpr std::array<int, numWindowTypes> windowOrders = {1, 1, 2};

std::string getEnvironmentVariable(const char*);
bool getEnvironmentVariableAsBool(const char*);
bool isDebugBuild();
float getPitch(int noteNumber);
float getCrotchetsPerSample(float crotchetsPerSecond, int samplesPerSecond);
std::vector<float> getAnalysisWindow(int windowSize, WindowType type);

constexpr float FastLog2(float x) {
    static_assert(sizeof(float) == sizeof(int32_t));
    union {
        float val;
        int32_t x;
    } u = {x};
    auto log_2 = (float)(((u.x >> 23) & 255) - 128);
    u.x &= ~(255 << 23);
    u.x += 127 << 23;
    log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val - 0.65871759316667f;
    return log_2;
}

template <typename F>
struct Finally {
    Finally(F f) : func(std::move(f)) {}
    ~Finally() {
        func();
    }
    F func;
};
}  // namespace utils
}  // namespace saint
