#pragma once

#include <functional>
#include <memory>

#include "PitchDetector.h"
#include "Recording/IRecordingListener.h"
#include "Recording/PitchDetectorRecording.h"

namespace saint {
/**
 * @brief Decorator that records the audio fed to a `PitchDetector` so that a
 * misbehaving live session can be replayed offline (see `ReplayPitchDetector`).
 *
 * The inner detector must be freshly constructed: the recording starts at the
 * detector's initial state, so a replay reproduces the live estimates exactly.
 * The recording completes after `durationSeconds` of audio, or earlier if
 * `stop()` is called. On completion, `OnComplete` is called, handing back the
 * inner detector - so that the caller can continue feeding it without a state
 * reset - together with the recorded data. The caller is responsible for
 * persisting the data, which must be done with `recording::writeWavFile` so
 * that the metadata needed by `ReplayPitchDetector` is written; being in
 * charge, the caller can decide whether the file writing may happen on the
 * audio thread or has to be offloaded onto another one.
 *
 * The remaining recording time is reported to the `IRecordingListener` (see
 * `IRecordingListener::onProgress`).
 *
 * The buffer is pre-allocated at construction; `process()` does not allocate.
 * `process()` must not be called anymore once the recording completed.
 */
class RecordingPitchDetector : public PitchDetector {
   public:
    using OnComplete =
        std::function<void(std::unique_ptr<PitchDetector> inner, recording::RecordingData)>;

    RecordingPitchDetector(std::unique_ptr<PitchDetector> inner, recording::PitchDetectorConfig,
                           int durationSeconds, IRecordingListener&, OnComplete);

    float process(const float* input, DebugOutput* = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

    /**
     * @brief Terminates the recording early, firing `OnComplete` with the blocks recorded so far.
     * No-op if the recording already completed.
     */
    void stop();

   private:
    void complete();

    const recording::PitchDetectorConfig _config;
    std::unique_ptr<PitchDetector> _inner;
    IRecordingListener& _listener;
    const OnComplete _onComplete;
    const int _samplesPerBlock;
    const int _maxNumBlocks;
    std::vector<float> _buffer;
    int _numBlocksStored = 0;
    int _lastReportedRemainingSeconds = -1;
};
}  // namespace saint
