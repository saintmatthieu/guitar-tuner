#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <numeric>

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
std::vector<float> getCoefs(utils::WindowType type) {
    switch (type) {
        case utils::WindowType::Hann:
            return {1.f, -1.f};
        case utils::WindowType::Hamming:
            // Found in PhD thesis Matthieu Hodgkinson @NUIM
            // https://mural.maynoothuniversity.ie/id/eprint/3910/1/thesis.pdf
            // Section 2.2.3, p90
            return {1.f, -349.f / 407.f};
        case utils::WindowType::MinimumThreeTerm:
            // Same
            return {1.f, -1152.f / 983.f, +515.f / 2792.f};
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

    const auto coefs = getCoefs(type);
    auto sum = 0.f;
    for (auto i = 0u; i < windowSize; ++i) {
        // i + 1 so that the tip of the window is at windowSize / 2, which is
        // convenient when taking the second half of it.
        window[i] = coefs[0];
        for (size_t j = 1; j < coefs.size(); ++j) {
            window[i] += coefs[j] * cosf((i + 1) * j * freq);
        }
        sum += window[i];
    }
    std::transform(window.begin(), window.end(), window.begin(),
                   [sum](float x) { return x / sum; });
    return window;
}

void utils::getDbSpectrum(const std::vector<std::complex<float>>& spectrum, float* out, int count) {
    if (count <= 0) {
        return;
    }

    // First bin is DC only.
    out[0] = utils::FastDb(spectrum[0].real() * spectrum[0].real());
    std::transform(spectrum.data() + 1, spectrum.data() + count, out + 1,
                   [&](const std::complex<float>& X) {
                       const auto power = X.real() * X.real() + X.imag() * X.imag();
                       return utils::FastDb(power);
                   });
}

float utils::quadFit(const float* y) {
    // vertex at x = 0.5 * (y[-1] - y[1]) / (y[-1] - 2 * y[0] + y[1])
    const auto delta = 0.5f * (y[0] - y[2]) / (y[2] - 2 * y[1] + y[0]);
    return delta;
}

float utils::getApproximateGcd(const std::vector<float>& values) {
    if (values.size() < 2) {
        return 0.f;
    }

    // Use a histogram approach to find the approximate GCD.
    const float minValue = *std::min_element(values.begin(), values.end());
    const float maxValue = *std::max_element(values.begin(), values.end());
    const int numBins = 100;
    std::vector<int> histogram(numBins, 0);
    const float binSize = (maxValue - minValue) / numBins;

    for (const auto value : values) {
        for (float factor = 1.f; factor * value <= maxValue; factor += 1.f) {
            const float scaledValue = factor * value;
            const int binIndex = static_cast<int>((scaledValue - minValue) / binSize);
            if (binIndex >= 0 && binIndex < numBins) {
                histogram[binIndex]++;
            }
        }
    }

    // Find the bin with the maximum count.
    int maxCount = 0;
    int bestBinIndex = 0;
    for (int i = 0; i < numBins; ++i) {
        if (histogram[i] >= maxCount) {
            maxCount = histogram[i];
            bestBinIndex = i;
        }
    }

    // The approximate GCD is the center of the best bin.
    return minValue + (bestBinIndex + 0.5f) * binSize;
}

namespace {
std::vector<int> getHighestLocalMaxima(const float* dbValues, size_t size, int offset, int N) {
    std::vector<int> localMaxima;
    const int intSize = static_cast<int>(size);
    for (int i = 1 + offset; i < intSize - 1; ++i) {
        // Also check that the absolute value is larger than -60dB (may need some tuning)
        if (dbValues[i] > -60.f && dbValues[i] > dbValues[i - 1] && dbValues[i] > dbValues[i + 1]) {
            localMaxima.push_back(i);
        }
    }
    std::sort(localMaxima.begin(), localMaxima.end(),
              [dbValues](int a, int b) { return dbValues[a] > dbValues[b]; });
    if (localMaxima.size() > static_cast<size_t>(N)) {
        localMaxima.resize(N);
    }
    return localMaxima;
}
}  // namespace

std::pair<float, float> utils::polyFit(const std::vector<float>& x, const std::vector<float>& y,
                                       const std::vector<float>& weights) {
    assert(x.size() == y.size());
    const size_t n = x.size();
    std::vector<float> w = weights;
    ;
    if (w.size() != n) {
        w.resize(n, 1.f);
    }
    float S = 0.f;
    float Sx = 0.f;
    float Sy = 0.f;
    float Sxx = 0.f;
    float Sxy = 0.f;
    for (size_t i = 0; i < n; ++i) {
        const float wi = w[i];
        S += wi;
        Sx += wi * x[i];
        Sy += wi * y[i];
        Sxx += wi * x[i] * x[i];
        Sxy += wi * x[i] * y[i];
    }
    const float denom = S * Sxx - Sx * Sx;
    if (denom == 0.f) {
        return {0.f, 0.f};
    }
    const float a = (S * Sxy - Sx * Sy) / denom;
    const float b = (Sxx * Sy - Sx * Sxy) / denom;
    return {a, b};
}

float utils::doubleCheckEstimate(float priorIndex, const std::vector<float>& dbSpectrum, int minBin,
                                 int N) {
    // clang-format off

    // `priorIndex` is likely correct but could also be the actual value by 2, 3 or 4 (harmonics
    // being interpreted as fundamental). Disambiguate this:
    // * Take the `N` (TBD) top peaks, ignoring anything less than `minBin`.
    // * Calculate a vector of weights. They must be based on the dB values, not the linear ones, to account for
    //   human perception.
    // For each 4 hypotheses (including the one that `prior` is correct):
    // * Let the hypothesis fundamental frequency be f0,
    // * derive the harmonic number `k` of each peak: k = f / f0
    // * estimate the error vector of size `N`: e(i) = k(i) - round(k(i))
    // * store the weighted sum of squares of this vector.
    // Return the most likely hypothesis.

    // clang-format on

    const std::vector<int> localMaxima =
        getHighestLocalMaxima(dbSpectrum.data(), dbSpectrum.size(), minBin, N);

    std::vector<float> weights;
    weights.reserve(localMaxima.size());
    for (int lm : localMaxima) {
        // Map dBs to linear domain
        const auto w = std::max(dbSpectrum[lm] / 60.f + 1.f, 0.f);
        weights.push_back(w);
    }

    float bestScore = std::numeric_limits<float>::max();
    float bestEstimate = priorIndex;
    for (auto divisor = 1; divisor <= 4; ++divisor) {
        const auto f0 = static_cast<float>(priorIndex) / divisor;
        std::vector<float> errors(localMaxima.size());
        std::transform(localMaxima.begin(), localMaxima.end(), errors.begin(), [f0](int peakBin) {
            const auto k = peakBin / f0;
            const auto kRounded = std::round(k);
            const auto e = k - kRounded;
            return e;
        });
        const auto score =
            std::inner_product(errors.begin(), errors.end(), weights.begin(), 0.f, std::plus<>(),
                               [](float e, float w) { return w * e * e; });
        if (score < bestScore) {
            bestScore = score;
            bestEstimate = f0;
        }
    }

    return bestEstimate;
}

float utils::doubleCheckEstimate(float prior, const std::vector<float>& dbSpectrum, int sampleRate,
                                 int fftSize, float minFreq) {
    const auto binFreq = static_cast<float>(sampleRate) / fftSize;
    const auto minBin = static_cast<int>(minFreq / binFreq + .5f);
    const auto estimate = doubleCheckEstimate(prior / binFreq, dbSpectrum, minBin, 5);
    return estimate * sampleRate / fftSize;
}

int utils::getIndexOfClosestLocalMaximum(const std::vector<float>& values, int startIndex) {
    // go up
    auto i = startIndex;
    while (i + 1 < values.size() && values[i + 1] >= values[i]) {
        ++i;
    }
    // go down
    while (i - 1 >= 0 && values[i - 1] > values[i]) {
        --i;
    }
    return i;
}
}  // namespace saint