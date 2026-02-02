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

void utils::getLogSpectrum(const std::vector<std::complex<float>>& spectrum, float* out,
                           int count) {
    if (count <= 0) {
        return;
    }

    // First bin is DC only.
    out[0] = utils::FastLog2(spectrum[0].real() * spectrum[0].real());
    auto k = 1;
    std::transform(spectrum.data() + 1, spectrum.data() + count, out + 1,
                   [&](const std::complex<float>& X) {
                       const auto power = X.real() * X.real() + X.imag() * X.imag();
                       return utils::FastLog2(power);
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
    // The GCD can't be larger than the smallest value in the set.
    const float maxValue = *std::min_element(values.begin(), values.end()) * 1.1f;
    const int numBins = 1000;
    std::vector<int> histogram(numBins, 0);
    const float binSize = maxValue / numBins;

    for (const auto value : values) {
        for (float factor = 1.f; factor * value <= maxValue; factor += 1.f) {
            const float scaledValue = factor * value;
            const int binIndex = static_cast<int>(scaledValue / binSize);
            if (binIndex >= 0 && binIndex < numBins) {
                histogram[binIndex]++;
            }
        }
    }

    // Find the bin with the maximum count.
    int maxCount = 0;
    int bestBinIndex = 0;
    for (int i = 0; i < numBins; ++i) {
        if (histogram[i] > maxCount) {
            maxCount = histogram[i];
            bestBinIndex = i;
        }
    }

    // The approximate GCD is the center of the best bin.
    return (bestBinIndex + 0.5f) * binSize;
}

std::optional<float> utils::estimateFundamentalByPeakPicking(std::vector<float> logSpectrum,
                                                             int sampleRate, int fftSize,
                                                             float minFreq, float maxFreq) {
    // Strategy:

    // Find the `P` highest peaks, not below `minFreq`, but searching harmonics beyond `maxFreq` is
    // allowed (`maxFreq` being the maximum fundamental frequency estimate).
    // Once the peaks have been found, look for the greatest common divisor.

    // const auto binFreq = static_cast<float>(sampleRate) / fftSize;
    // const auto minBin = static_cast<int>(std::ceil(minFreq / binFreq));
    // const auto maxBin = std::min(static_cast<int>(logSpectrum.size() - 1),
    //                              static_cast<int>(std::floor(maxFreq / binFreq)));

    // constexpr auto P = 5;
    // std::vector<std::vector<float>::const_iterator> its;
    // its.reserve(P);

    return std::nullopt;
}
}  // namespace saint