#pragma once

#include <memory>
#include <optional>

namespace saint {
enum class PitchClass { C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B };

struct Pitch {
    const PitchClass pitchClass;
    const int octave;
};

class PitchDetector {
   public:
    struct Config {
        const std::optional<Pitch> lowestPitch;
        const std::optional<Pitch> highestPitch;
    };

    static std::unique_ptr<PitchDetector> createInstance(
        int sampleRate, int blockSize, const std::optional<Config>& config = std::nullopt);

    static constexpr auto maxBlockSize = 8192;
    /**
     * @brief Processes a block of audio samples and return the detected pitch in
     * Hz.
     *
     * @param input pointer to exactly `blockSize` samples (as specified at construction)
     * @param presenceScore if not null, on return contains a value between 0 and
     * 1 indicating the confidence that a pitch is present in the audio. Meant for
     * evaluation.
     * @return float 0 if no pitch detected, the value in Hz if pitch is detected.
     */
    virtual float process(const float* input, float* presenceScore = nullptr) = 0;
    virtual ~PitchDetector() = default;
};
}  // namespace saint