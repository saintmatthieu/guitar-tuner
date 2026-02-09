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

constexpr bool isPowerOfTwo(int x) {
    return (x & (x - 1)) == 0 && x > 0;
}
static_assert(isPowerOfTwo(2));
static_assert(!isPowerOfTwo(3));
static_assert(isPowerOfTwo(1024));
static_assert(!isPowerOfTwo(1024 + 96));

/**
 * @brief Checks for symmetry in a hermitian sense: vec[i] == vec[size - i]
 * Only valid for containers of non-complex types.
 */
template <typename T>
constexpr bool isSymmetric(const T& vec) {
    for (size_t i = 1; i < vec.size() / 2; ++i) {
        if (std::complex(vec[i]) != vec[vec.size() - i]) {
            return false;
        }
    }
    return true;
}
static_assert(isSymmetric(std::array<int, 8>{0, 1, 2, 3, 4, 3, 2, 1}));

/**
 * @brief Get full power spectrum (including symmetric part) from the half-spectrum returned by the
 * FFT.
 */
void getPowerSpectrum(const std::vector<std::complex<float>>& spectrum, std::vector<float>& out);

float getApproximateGcd(const std::vector<float>& values);

/**
 * @brief assuming y to be of size 3 with y[1] the maximum and the x coordinates -1, 0, 1,
 * return the x coordinate of the vertex of the parabola fitting the three points.
 *
 * @param y
 * @return float
 */
float quadFit(const float* y);

std::pair<float, float> lineFit(const std::vector<float>& x, const std::vector<float>& y,
                                const std::vector<float>& weights = {});

template <typename T, typename U>
constexpr typename U::value_type leastSquareFit(const T& x, const U& y, U w = {}) {
    if (w.size() != y.size()) {
        w.resize(y.size(), 1.f);
    }
    typename U::value_type num = 0;
    typename U::value_type den = 0;
    for (auto i = 0; i < x.size(); ++i) {
        num += x[i] * y[i] * w[i];
        den += x[i] * x[i] * w[i];
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
