#pragma once

#include <vector>

#include "PitchDetectorTypes.h"
#include "Utils/CommonTypes.h"

namespace saint {
constexpr auto kWindowType = WindowType::MinimumThreeTerm;

class PitchDetector {
   public:
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
     * @return float 0 if no pitch detected, the value in Hz if pitch is detected.
     */
    virtual float process(const float* input, DebugOutput* = nullptr,
                          std::vector<float>* debugOutputSignal = nullptr) = 0;
    virtual int delaySamples() const = 0;
    virtual ~PitchDetector() = default;
};
}  // namespace saint