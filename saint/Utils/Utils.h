#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <optional>
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

constexpr float FastDb(float power) {
    const auto log10 = FastLog2(power) / 3.321928094887362f;
    return 10.f * log10;
}

/**
 * @param count `spectrum` and `out` have the same FFT format, but only the first `count`
 * bins are transformed, to save computation.
 */
void getDbSpectrum(const std::vector<std::complex<float>>& spectrum, std::vector<float>& out,
                   int count = -1);

float getApproximateGcd(const std::vector<float>& values);

/**
 * @brief assuming y to be of size 3 with y[1] the maximum and the x coordinates -1, 0, 1,
 * return the x coordinate of the vertex of the parabola fitting the three points.
 *
 * @param y
 * @return float
 */
float quadFit(const float* y);

std::pair<float, float> polyFit(const std::vector<float>& x, const std::vector<float>& y,
                                const std::vector<float>& weights = {});

template <typename T, typename U>
double leastSquareFit(const T& x, const U& y) {
    auto num = 0.;
    auto den = 0.;
    for (auto i = 0; i < x.size(); ++i) {
        num += x[i] * y[i];
        den += x[i] * x[i];
    }
    const auto a = num / den;

    return a;
}

float doubleCheckEstimate(float priorFreq, const std::vector<float>& dbSpectrum, int sampleRate,
                          int fftSize, float minFreq);

int getIndexOfClosestLocalMaximum(const std::vector<float>& values, int startIndex);

/**
 * @brief Same as above, but using bins - easier for testing.
 *
 * @param priorIndex still a floating-point, even though in index domain.
 */
float doubleCheckEstimate(float priorIndex, const std::vector<float>& dbSpectrum, int minBin,
                          int N);

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
