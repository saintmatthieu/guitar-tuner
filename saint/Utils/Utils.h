#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "CommonTypes.h"

namespace saint {
namespace utils {
std::string getEnvironmentVariable(const char*);
bool getEnvironmentVariableAsBool(const char*);
bool isDebugBuild();
float getPitch(int noteNumber);
float getCrotchetsPerSample(float crotchetsPerSecond, int samplesPerSecond);

// Source - https://stackoverflow.com/a/11575574
// Posted by njuffa, modified by community. See post 'Timeline' for change history
// Retrieved 2026-02-28, License - CC BY-SA 3.0

/* not quite rint(), i.e. results not properly rounded to nearest-or-even */
inline double my_rint(double x) {
    double t = floor(fabs(x) + 0.5);
    return (x < 0.0) ? -t : t;
}

/* minimax approximation to cos on [-pi/4, pi/4] with rel. err. ~= 7.5e-13 */
inline double cos_core(double x) {
    double x8, x4, x2;
    x2 = x * x;
    x4 = x2 * x2;
    x8 = x4 * x4;
    /* evaluate polynomial using Estrin's scheme */
    return (-2.7236370439787708e-7 * x2 + 2.4799852696610628e-5) * x8 +
           (-1.3888885054799695e-3 * x2 + 4.1666666636943683e-2) * x4 +
           (-4.9999999999963024e-1 * x2 + 1.0000000000000000e+0);
}

/* minimax approximation to sin on [-pi/4, pi/4] with rel. err. ~= 5.5e-12 */
inline double sin_core(double x) {
    double x4, x2, t;
    x2 = x * x;
    x4 = x2 * x2;
    /* evaluate polynomial using a mix of Estrin's and Horner's scheme */
    return ((2.7181216275479732e-6 * x2 - 1.9839312269456257e-4) * x4 +
            (8.3333293048425631e-3 * x2 - 1.6666666640797048e-1)) *
               x2 * x +
           x;
}

/* minimax approximation to arcsin on [0, 0.5625] with rel. err. ~= 1.5e-11 */
inline double asin_core(double x) {
    double x8, x4, x2;
    x2 = x * x;
    x4 = x2 * x2;
    x8 = x4 * x4;
    /* evaluate polynomial using a mix of Estrin's and Horner's scheme */
    return (((4.5334220547132049e-2 * x2 - 1.1226216762576600e-2) * x4 +
             (2.6334281471361822e-2 * x2 + 2.0596336163223834e-2)) *
                x8 +
            (3.0582043602875735e-2 * x2 + 4.4630538556294605e-2) * x4 +
            (7.5000364034134126e-2 * x2 + 1.6666666300567365e-1)) *
               x2 * x +
           x;
}

/* relative error < 7e-12 on [-50000, 50000] */
inline double my_sin(double x) {
    double q, t;
    int quadrant;
    /* Cody-Waite style argument reduction */
    q = my_rint(x * 6.3661977236758138e-1);
    quadrant = (int)q;
    t = x - q * 1.5707963267923333e+00;
    t = t - q * 2.5633441515945189e-12;
    if (quadrant & 1) {
        t = cos_core(t);
    } else {
        t = sin_core(t);
    }
    return (quadrant & 2) ? -t : t;
}

template <WindowType W>
constexpr int mainLobeWidth() {
    return windowOrders.at(static_cast<size_t>(W)) * 2 + 1;
}

template <typename T = float>
std::vector<T> getCoefs(WindowType type) {
    switch (type) {
        case WindowType::Rectangular:
            return {static_cast<T>(1)};
        case WindowType::Hann:
            return {static_cast<T>(1), static_cast<T>(-1)};
        case WindowType::Hamming:
            // Found in PhD thesis Matthieu Hodgkinson @NUIM
            // https://mural.maynoothuniversity.ie/id/eprint/3910/1/thesis.pdf
            // Section 2.2.3, p90
            return {static_cast<T>(1), static_cast<T>(-349) / 407};
        case WindowType::MinimumThreeTerm:
            // Same
            return {static_cast<T>(1), static_cast<T>(-1152) / 983, static_cast<T>(515) / 2792};
        default:
            assert(false);
            return getCoefs<T>(WindowType::Hann);
    }
}

inline double sinc(double x) {
    if (std::abs(x) < 1e-8) {
        return 1.;
    } else {
        const auto pix = M_PI * x;
        return my_sin(pix) / pix;
    }
}

/**
 * @brief Evaluate main lobe of window `W` at continuous bin `b`.
 */
template <WindowType W>
double mainLobeAt(double b) {
    constexpr auto P = windowOrders[static_cast<int>(W)];
    const auto a = getCoefs<double>(W);
    // Convolution of sinc with the frequency components of the window.
    // The first component is that of the rectangular window.
    auto sum = std::abs(a[0]) * sinc(b);
    for (auto p = 1; p <= P; ++p) {
        // Then for new cosine component of the window, since it's a real signal, shift-multiply
        // with an impulse at that frequency and at the negative of that frequency.
        const auto left = sinc(b - p);
        const auto right = sinc(b + p);
        sum += std::abs(a[p]) * (left + right) / 2;
    }
    return sum;
}

template <typename T = float>
std::vector<T> getAnalysisWindow(int windowSize, WindowType type) {
    std::vector<T> window((size_t)windowSize);
    constexpr T twoPi = 6.283185307179586;
    const T freq = twoPi / windowSize;

    const auto coefs = getCoefs<T>(type);
    T sum = 0;
    for (auto i = 0u; i < windowSize; ++i) {
        // i + 1 so that the tip of the window is at windowSize / 2, which is
        // convenient when taking the second half of it.
        window[i] = coefs[0];
        for (size_t j = 1; j < coefs.size(); ++j) {
            window[i] += coefs[j] * cosf((i + 1) * j * freq);
        }
        sum += window[i];
    }
    std::transform(window.begin(), window.end(), window.begin(), [sum](T x) { return x / sum; });
    return window;
}

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
float quadFit(const float* y, float* out = nullptr);

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
