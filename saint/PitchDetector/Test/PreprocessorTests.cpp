#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "Preprocessor.h"
#include "testUtils.h"

namespace saint {
TEST(Preprocessor, process) {
    // Create 1 second of white noise at 44.1kHz
    constexpr auto format = ChannelFormat::Mono;
    const int sampleRate = 44100;
    const int numChannels = static_cast<int>(format);
    const int numFrames = sampleRate;
    std::vector<float> audio(numFrames * numChannels);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& sample : audio) {
        sample = dist(rng);
    }

    Preprocessor preprocessor(sampleRate, numChannels);
    preprocessor.process(audio.data(), numFrames);

    testUtils::toWavFile(testUtils::getOutDir() / "preprocessor_test_output.wav",
                         testUtils::Audio{std::move(audio), sampleRate, format}, nullptr,
                         "PreprocessorTest");
}
}  // namespace saint