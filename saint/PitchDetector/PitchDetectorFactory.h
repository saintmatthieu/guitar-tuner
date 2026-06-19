#pragma once

#include <functional>
#include <memory>
#include <string>

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
 * @param holdPitch When true (default), the detector briefly holds the last pitch through a
 * presence dip so the UI does not blink (see `PitchDetectorConfig::holdPitch`). Pass false to
 * emit 0 the moment detection drops.
 * @param cpuSummaryCallback Forwarded to the `IssueReportingPitchDetector`, which invokes it on
 * destruction with a one-line CPU-load summary (see its destructor). Optional.
 */
std::unique_ptr<IssueReportingPitchDetector> createInstance(
    int sampleRate, ChannelFormat, int samplesPerBlockPerChannel, Tuning tuning = Tuning::Standard,
    bool holdPitch = true, std::function<void(std::string logLine)> cpuSummaryCallback = {});
}  // namespace PitchDetectorFactory
}  // namespace saint
