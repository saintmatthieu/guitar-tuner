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
        int sampleRate, const std::optional<Config>& config = std::nullopt);

    static constexpr auto maxBlockSize = 8192;
    /**
     * @brief Processes a block of audio samples and return the detected pitch in
     * Hz.
     *
     * @param input
     * @param numSamples not to exceed maxBlockSize
     * @param presenceScore if not null, on return contains a value between 0 and
     * 1 indicating the confidence that a pitch is present in the audio. Meant for
     * evaluation.
     * @return std::optional<float> 0 if no pitch detected, the value in Hz if
     * pitch is detected, and nullopt if it needs more audio to provide an update.
     * Note: if `numSamples` is so big that several updates can be made, only the
     * last one is returned.
     */
    virtual std::optional<float> process(const float* input, int numSamples,
                                         float* presenceScore = nullptr) = 0;
    virtual ~PitchDetector() = default;
};
}  // namespace saint