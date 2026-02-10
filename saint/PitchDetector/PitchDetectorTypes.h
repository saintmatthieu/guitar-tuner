#pragma once

#include <optional>

namespace saint {
enum class PitchClass { C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B, OneKiloHz /*for testing*/ };

struct Pitch {
    const PitchClass pitchClass;
    const int octave;
};

enum class ChannelFormat { Mono, Stereo };

struct PitchDetectorConfig {
    const std::optional<Pitch> lowestPitch;
    const std::optional<Pitch> highestPitch;
};
}  // namespace saint