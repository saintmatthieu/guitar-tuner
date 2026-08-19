#pragma once

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "PitchDetectorTypes.h"

namespace saint {
enum class PitchBucket {
    belowRange,
    inRange,
    aboveRange,
};

/**
 * @brief Return value of @ref PitchDetector::process. `bucket == nullopt` if and only if `pitch ==
 * 0`.
 */
struct PitchDetectionResult {
    float pitch = 0.f;
    std::optional<PitchBucket> bucket;
};

/**
 * @brief Classifies `frequency` against a @ref PitchDetector::pitchSearchRange.
 */
inline PitchBucket getBucket(float frequency, const std::pair<float, float>& searchRange) {
    return frequency < searchRange.first    ? PitchBucket::belowRange
           : frequency > searchRange.second ? PitchBucket::aboveRange
                                            : PitchBucket::inRange;
}

class PitchDetector {
   public:
    /**
     * @details Encompasses the tuning range (see the `Tuning` argument of @ref
     * PitchDetectorFactory::createInstance), plus implementation-specific margins. Helpful to
     * figure out the precise meaning of the `PitchBucket` values.
     */
    virtual std::pair<float, float> pitchSearchRange() const {
        return {0.f, std::numeric_limits<float>::max()};
    }

    /**
     * @brief Processes a block of audio samples and return the detected pitch in
     * Hz.
     *
     * @param input pointer to exactly `samplesPerBlockPerChannel * numChannels` samples (as
     * specified at construction), interleaved if stereo.
     * @param presenceScore FOR TESTING - if not null, on return contains a value between 0 and
     * 1 indicating the confidence that a pitch is present in the audio.
     * @param debugOutputSignal FOR TESTING - if not null, internally pre-processed signal with be
     * appended.
     */
    virtual PitchDetectionResult process(const float* input, DebugOutput* = nullptr,
                                         std::vector<float>* debugOutputSignal = nullptr) = 0;
    virtual int delaySamples() const = 0;
    virtual ~PitchDetector() = default;
};
}  // namespace saint