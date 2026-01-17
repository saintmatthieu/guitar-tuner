#pragma once

#include <memory>
#include <optional>

namespace saint {
class PitchDetector {
public:
  static std::unique_ptr<PitchDetector>
  createInstance(int sampleRate,
                 const std::optional<float> &leastFrequencyToDetect);
  static constexpr auto maxBlockSize = 8192;
  virtual std::optional<float> process(const float *, int) = 0;
  virtual ~PitchDetector() = default;
};
} // namespace saint