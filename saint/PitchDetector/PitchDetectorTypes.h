#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace saint {
using DebugOutput = std::unordered_map<std::string, float>;

enum class PitchClass { C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B, OneKiloHz /*for testing*/ };

static constexpr auto autocorrCutoffFreqHz = 1500;
static constexpr auto autocorrRolloffHz = 200;
// Assuming that we're only dealing with 6-string guitars in standard tuning and standard sample
// rates, the worst case of quantization when estimating the frequency from reading the
// autocorrelation is an E4 at 44.1kHz.
// Upsampling by a factor of 4, we reduce the maximal quantization to 1.5 cents.
constexpr auto autocorrUpsamplingFactor = 4;

struct Pitch {
    const PitchClass pitchClass;
    const int octave;
};

enum class ChannelFormat { Mono = 1, Stereo = 2 };
constexpr int numChannels(ChannelFormat format) {
    return static_cast<int>(format);
}

struct PitchDetectorConfig {
    const std::optional<Pitch> lowestPitch;
    const std::optional<Pitch> highestPitch;
};
}  // namespace saint