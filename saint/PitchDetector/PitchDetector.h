#pragma once

#include <memory>
#include <optional>

namespace saint {
class PitchDetector {
public:
  static std::unique_ptr<PitchDetector> createInstance(int sampleRate);

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
  virtual std::optional<float> process(const float *input, int numSamples,
                                       float *presenceScore = nullptr) = 0;
  virtual ~PitchDetector() = default;
};
} // namespace saint