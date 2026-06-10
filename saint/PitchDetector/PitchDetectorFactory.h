#pragma once

#include <memory>

#include "PitchDetector.h"

namespace saint {
namespace PitchDetectorFactory {
std::unique_ptr<PitchDetector> createInstance(int sampleRate, ChannelFormat,
                                              int samplesPerBlockPerChannel,
                                              Tuning tuning = Tuning::Standard);
}  // namespace PitchDetectorFactory
}  // namespace saint