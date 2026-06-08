#pragma once

#include <memory>
#include <optional>

#include "FixedBlockPitchDetector.h"

namespace saint {
namespace PitchDetectorFactory {
std::unique_ptr<FixedBlockPitchDetector> createInstance(
    int sampleRate, ChannelFormat, int samplesPerBlockPerChannel,
    const std::optional<PitchDetectorConfig>& config = std::nullopt);
}  // namespace PitchDetectorFactory
}  // namespace saint