#pragma once

#include <atomic>
#include <functional>
#include <memory>

#include "PitchDetector.h"
#include "Recording/IRecordingListener.h"
#include "Recording/PitchDetectorRecording.h"
#include "Recording/RecordingPitchDetector.h"

namespace saint {
/**
 * @brief The `PitchDetector` returned by `PitchDetectorFactory::createInstance`: forwards the
 * audio to the real implementation, and lets the user report a misbehavior via
 * `startIssueRecording`.
 */
class IssueReportingPitchDetector : public PitchDetector {
   public:
    IssueReportingPitchDetector(recording::PitchDetectorConfig,
                                std::function<std::unique_ptr<PitchDetector>()> detectorFactory);

    float process(const float* input, DebugOutput* = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

    /**
     * @brief Starts recording the audio stream to diagnose an issue offline.
     * @details The implementation is exchanged with a recording one wrapping a fresh detector -
     * the recording starts at the detector's initial state, so that a replay (see
     * `ReplayPitchDetector`) reproduces the live estimates bit-exactly. After `durationSeconds`
     * of audio the recording is complete: the listener's `onComplete` receives the recorded
     * data, and processing continues with the recorder's detector - stopping the recording
     * doesn't cause a state reset, only starting it does. While the recording is ongoing, the
     * remaining time is reported via the listener's `onProgress`. If a recording is already in
     * progress, it is gracefully terminated - its listener's `onComplete` fires with the blocks
     * recorded so far - and the new one starts on the fly.
     *
     * The listener must outlive the recording. Persisting the data is up to the listener - see
     * `IRecordingListener::onComplete` for the requirements.
     *
     * This call allocates in one go the memory necessary for `durationSeconds` of audio -
     * acceptable, since the user just pressed a button and is likely not in the middle of a note
     * recording.
     */
    void startIssueRecording(int durationSeconds, IRecordingListener&);
    bool isRecording() const;

    /**
     * @brief CPU load of `process` calls, as a percentage of the audio frame duration
     * (100 * processingTime / frameDuration), smoothed by a first-order lowpass with a decay
     * time of 1 second. 0 until `process` was first called. May be called from any thread.
     */
    int realtimePercentage() const;

   private:
    const recording::PitchDetectorConfig _config;
    const std::function<std::unique_ptr<PitchDetector>()> _detectorFactory;
    std::unique_ptr<PitchDetector> _detector;
    std::unique_ptr<RecordingPitchDetector> _recorder;
    bool _recordingComplete = false;
    const double _frameDuration;
    const double _lowpassCoeff;
    double _smoothedPercentage = 0;
    std::atomic<int> _realtimePercentage = 0;
};
}  // namespace saint
