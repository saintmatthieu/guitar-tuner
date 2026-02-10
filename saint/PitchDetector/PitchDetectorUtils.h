#pragma once

#include <optional>

#include "PitchDetectorTypes.h"

namespace saint {
float getMinFreq(const std::optional<PitchDetectorConfig>& config);
float getMaxFreq(const std::optional<PitchDetectorConfig>& config);
}  // namespace saint