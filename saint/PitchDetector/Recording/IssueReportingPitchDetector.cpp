#include "IssueReportingPitchDetector.h"

#include <cassert>
#include <chrono>
#include <cmath>

namespace saint {
IssueReportingPitchDetector::IssueReportingPitchDetector(
    recording::PitchDetectorConfig config,
    std::function<std::unique_ptr<PitchDetector>()> detectorFactory)
    : _config(config),
      _detectorFactory(std::move(detectorFactory)),
      _frameDuration(static_cast<double>(config.samplesPerBlockPerChannel) / config.sampleRate),
      // One-pole lowpass whose step response reaches 0.9 after 1 second, i.e.
      // 1 - coeff^(1s / frameDuration) = 0.9.
      _lowpassCoeff(std::pow(0.1, _frameDuration)) {
    assert(_detectorFactory);
    _detector = _detectorFactory();
}

float IssueReportingPitchDetector::process(const float* input, DebugOutput* debugOutput,
                                           std::vector<float>* debugOutputSignal) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = [&] {
        if (_recorder) {
            const auto result = _recorder->process(input, debugOutput, debugOutputSignal);
            if (_recordingComplete) {
                // The recorder fired its completion callback during process(), handing its
                // detector back (see startIssueRecording); it is now empty and can be dropped.
                _recorder.reset();
                _recordingComplete = false;
            }
            return result;
        }
        return _detector->process(input, debugOutput, debugOutputSignal);
    }();
    const std::chrono::duration<double> processingTime = std::chrono::steady_clock::now() - start;
    const auto percentage = 100 * processingTime.count() / _frameDuration;
    _smoothedPercentage = _lowpassCoeff * _smoothedPercentage + (1 - _lowpassCoeff) * percentage;
    _realtimePercentage.store(static_cast<int>(std::lround(_smoothedPercentage)),
                              std::memory_order_relaxed);
    return result;
}

int IssueReportingPitchDetector::delaySamples() const {
    return _recorder ? _recorder->delaySamples() : _detector->delaySamples();
}

void IssueReportingPitchDetector::startIssueRecording(int durationSeconds,
                                                      IRecordingListener& listener) {
    if (_recorder) {
        // Gracefully terminate the ongoing recording: its listener's onComplete fires with the
        // blocks recorded so far. (The handed-back detector gets discarded right away, since the
        // new recording needs a fresh one anyway.)
        _recorder->stop();
        _recorder.reset();
        _recordingComplete = false;
    }
    _recorder = std::make_unique<RecordingPitchDetector>(
        _detectorFactory(), _config, durationSeconds, listener,
        [this, &listener](std::unique_ptr<PitchDetector> inner, recording::RecordingData data) {
            // From now on, use the handed-back detector, so that we avoid a state reset of the
            // algorithm.
            _detector = std::move(inner);
            _recordingComplete = true;
            listener.onComplete(std::move(data));
        });
    _detector.reset();
}

bool IssueReportingPitchDetector::isRecording() const {
    return _recorder != nullptr;
}

int IssueReportingPitchDetector::realtimePercentage() const {
    return _realtimePercentage.load(std::memory_order_relaxed);
}
}  // namespace saint
