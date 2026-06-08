#pragma once

#include <vector>

#include "PitchDetectorTypes.h"

namespace saint {
// Pitch detector that accepts an arbitrary number of samples per call, decoupling the caller from
// any internal block size. Implementations re-block as needed (see ReblockingPitchDetector), so
// real-time sources whose buffer size is neither constant nor equal to the analysis block size
// (iOS AVAudioEngine taps, Android AudioRecord short reads, Web Audio render quanta) can feed audio
// directly.
class PitchDetector {
   public:
    /**
     * @brief Processes `n` mono audio samples and returns the detected pitch in Hz.
     *
     * @param input pointer to `n` samples (interleaved if stereo); may be null when `n == 0`.
     * @param n number of samples at `input`; any size, including 0.
     * @param debugOutput FOR TESTING - if not null, on return reflects the last analysis block
     * processed during this call (e.g. "presenceScore", the confidence a pitch is present).
     * @param debugOutputSignal FOR TESTING - if not null, the internally pre-processed signal of
     * each analysis block processed during this call is appended.
     * @return float 0 if no pitch was detected (or no complete block was available this call),
     * otherwise the most recently completed block's pitch in Hz.
     */
    virtual float process(const float* input, int n, DebugOutput* debugOutput = nullptr,
                          std::vector<float>* debugOutputSignal = nullptr) = 0;
    virtual int delaySamples() const = 0;
    virtual ~PitchDetector() = default;
};
}  // namespace saint
