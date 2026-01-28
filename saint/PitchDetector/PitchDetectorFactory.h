#pragma once

#include <memory>
#include <optional>

#include "PitchDetector.h"

namespace saint {
namespace PitchDetectorFactory {
std::unique_ptr<PitchDetector> createInstance(
    int sampleRate, ChannelFormat, int samplesPerBlockPerChannel,
    const std::optional<PitchDetector::Config>& config = std::nullopt);
}  // namespace PitchDetectorFactory
}  // namespace saint