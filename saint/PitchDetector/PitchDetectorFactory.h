#pragma once

#include <memory>

#include "PitchDetector.h"

namespace saint {
class IssueReportingPitchDetector;

namespace PitchDetectorFactory {
/**
 * @brief Create a pitch detector.
 * @details The returned `IssueReportingPitchDetector` is a thin wrapper that forwards the audio
 * to the real implementation, and additionally offers `startIssueRecording()`: the first
 * x seconds of the audio stream as well as the necessary configuration get saved to a WAV file
 * that can be replayed later (see `replayMain.cpp`) to diagnose problems that might have
 * occurred in live.
 */
std::unique_ptr<IssueReportingPitchDetector> createInstance(int sampleRate, ChannelFormat,
                                                            int samplesPerBlockPerChannel,
                                                            Tuning tuning = Tuning::Standard);
}  // namespace PitchDetectorFactory
}  // namespace saint
