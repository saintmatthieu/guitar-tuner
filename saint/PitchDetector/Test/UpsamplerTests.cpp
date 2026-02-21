#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "PitchDetectorTypes.h"
#include "Upsampler.h"
#include "testUtils.h"

namespace saint {
TEST(Upsampler, process) {
    constexpr auto fs = 44100;
    constexpr auto f0 = 100;
    constexpr auto maxFreq = autocorrCutoffFreqHz + autocorrRolloffHz;
    constexpr int K = maxFreq / f0;
    constexpr auto twoPi = 6.28318530718;
    std::vector<float> input(fs, 0.f);
    for (auto k = 1; k <= K; ++k) {
        for (auto n = 0; n < input.size(); ++n) {
            input[n] += std::cos(1. * n * k * twoPi * f0 / fs) / K * 0.5;
        }
    }
    testUtils::toWavFile(testUtils::getOutDir() / "upsamplerTestInput.wav",
                         testUtils::Audio{input, fs}, nullptr, "Upsampler input");

    Upsampler upsampler(fs, 2, maxFreq);
    const auto output = upsampler.process(input.data(), input.size());
    testUtils::toWavFile(testUtils::getOutDir() / "upsamplerTestOutput.wav",
                         testUtils::Audio{output, fs * autocorrUpsamplingFactor}, nullptr,
                         "Upsampler output");
}
}  // namespace saint