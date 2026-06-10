#pragma once

#include "Recording/PitchDetectorRecording.h"

namespace saint {
/**
 * @brief Receives feedback about an ongoing issue recording (see
 * `IssueReportingPitchDetector::startIssueRecording`). Must outlive the recording.
 */
class IRecordingListener {
   public:
    virtual ~IRecordingListener() = default;

    /**
     * @brief Reports the time remaining until the recording completes.
     * @details Called from within `process()` whenever the (rounded-up) number of remaining
     * seconds changes - i.e. about once a second, the last time with 0 just before `onComplete`.
     */
    virtual void onProgress(int remainingSeconds) = 0;

    /**
     * @brief The recording is complete - either the maximum duration was reached or it was
     * gracefully terminated by the start of a new recording.
     * @details The listener is responsible for persisting the data, which must be done with
     * `recording::writeWavFile` - and not a generic WAV writer - so that the metadata needed by
     * `ReplayPitchDetector` is written. Being in charge, the listener can decide whether the
     * file writing may happen on the audio thread or has to be offloaded onto another one.
     */
    virtual void onComplete(recording::RecordingData) = 0;
};
}  // namespace saint
