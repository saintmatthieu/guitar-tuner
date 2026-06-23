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
 * @param cpuSummaryCallback Forwarded to the `IssueReportingPitchDetector`, which invokes it on
 * destruction with a one-line CPU-load summary (see its destructor). Optional.
 * @param applyConstraintBandPass When true, the autocorrelation is band-passed around the locked
 * fundamental in the decaying tail (see autocorrConstraintBandHalfWidthSemitones). Off by default
 * (the shipping behaviour); the TestApp enables it so it can be auditioned against live input.
 */
std::unique_ptr<IssueReportingPitchDetector> createInstance(
    int sampleRate, ChannelFormat, int samplesPerBlockPerChannel, Tuning tuning = Tuning::Standard,
    std::function<void(std::string logLine)> cpuSummaryCallback = {},
    bool applyConstraintBandPass = false);
}  // namespace PitchDetectorFactory
}  // namespace saint
