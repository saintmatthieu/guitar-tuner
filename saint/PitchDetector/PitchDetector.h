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
   * @return std::optional<float> nullopt if no pitch detected, the value in Hz
   * otherwise.
   */
  virtual std::optional<float> process(const float *input, int numSamples) = 0;
  virtual ~PitchDetector() = default;
};
} // namespace saint