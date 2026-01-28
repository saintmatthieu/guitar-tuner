#pragma once

#include <memory>
#include <optional>

#include "PitchDetector.h"

namespace saint {
namespace PitchDetectorFactory {
std::unique_ptr<PitchDetector> createInstance(
    int sampleRate, int blockSize,
    const std::optional<PitchDetector::Config>& config = std::nullopt);
}
}  // namespace saint