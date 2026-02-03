#pragma once

#include <memory>
#include <optional>

namespace saint {
enum class PitchClass { C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B, OneKiloHz /*for testing*/ };

struct Pitch {
    const PitchClass pitchClass;
    const int octave;
};

enum class ChannelFormat { Mono, Stereo };

class PitchDetector {
   public:
    struct Config {
        const std::optional<Pitch> lowestPitch;
        const std::optional<Pitch> highestPitch;
    };

    /**
     * @brief Processes a block of audio samples and return the detected pitch in
     * Hz.
     *
     * @param input pointer to exactly `samplesPerBlockPerChannel * numChannels` samples (as
     * specified at construction), interleaved if stereo.
     * @param presenceScore if not null, on return contains a value between 0 and
     * 1 indicating the confidence that a pitch is present in the audio. Meant for
     * evaluation.
     * @return float 0 if no pitch detected, the value in Hz if pitch is detected.
     */
    virtual float process(const float* input, float* presenceScore = nullptr) = 0;
    virtual int delaySamples() const = 0;
    virtual ~PitchDetector() = default;
};
}  // namespace saint