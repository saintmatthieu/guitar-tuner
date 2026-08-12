#include "IssueReportingPitchDetector.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <sstream>

namespace saint {
IssueReportingPitchDetector::IssueReportingPitchDetector(
    recording::PitchDetectorConfig config,
    std::function<std::unique_ptr<PitchDetector>()> detectorFactory,
    std::function<void(std::string logLine)> cpuSummaryCallback)
    : _config(config),
      _detectorFactory(std::move(detectorFactory)),
      _cpuSummaryCallback(std::move(cpuSummaryCallback)),
      _frameDuration(static_cast<double>(config.samplesPerBlockPerChannel) / config.sampleRate),
      // One-pole lowpass whose step response reaches 0.9 after 1 second, i.e.
      // 1 - coeff^(1s / frameDuration) = 0.9.
      _lowpassCoeff(std::pow(0.1, _frameDuration)),
      _blocksPerSecond(config.sampleRate / config.samplesPerBlockPerChannel) {
    assert(_detectorFactory);
    _detector = _detectorFactory();
}

IssueReportingPitchDetector::~IssueReportingPitchDetector() {
    if (!_cpuSummaryCallback || _cpuSampleCount == 0) {
        return;
    }
    const auto avg = _cpuSampleSum / static_cast<double>(_cpuSampleCount);
    std::ostringstream line;
    line << "CPU load over " << _cpuSampleCount << " s: avg " << std::lround(avg) << "%, min "
         << _cpuSampleMin << "%, max " << _cpuSampleMax << "%";
    _cpuSummaryCallback(line.str());
}

PitchDetectionResult IssueReportingPitchDetector::process(const float* input,
                                                           DebugOutput* debugOutput,
                                                           std::vector<float>* debugOutputSignal) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = [&]() -> PitchDetectionResult {
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
    const auto rounded = static_cast<int>(std::lround(_smoothedPercentage));
    _realtimePercentage.store(rounded, std::memory_order_relaxed);
    if (++_blocksSinceLastSample >= _blocksPerSecond) {
        _cpuSampleMin = _cpuSampleCount == 0 || rounded < _cpuSampleMin ? rounded : _cpuSampleMin;
        _cpuSampleMax = _cpuSampleCount == 0 || rounded > _cpuSampleMax ? rounded : _cpuSampleMax;
        _cpuSampleSum += rounded;
        ++_cpuSampleCount;
        _blocksSinceLastSample = 0;
    }
    return result;
}

int IssueReportingPitchDetector::delaySamples() const {
    return _recorder ? _recorder->delaySamples() : _detector->delaySamples();
}

std::pair<float, float> IssueReportingPitchDetector::pitchSearchRange() const {
    return _recorder ? _recorder->pitchSearchRange() : _detector->pitchSearchRange();
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

    auto recordingFinishedCallback = [this, &listener](std::unique_ptr<PitchDetector> inner,
                                                       recording::RecordingData data) {
        // From now on, use the handed-back detector, so that we avoid a state reset of the
        // algorithm.
        _detector = std::move(inner);
        _recordingComplete = true;
        listener.onComplete(std::move(data));
    };

    _recorder =
        std::make_unique<RecordingPitchDetector>(_detectorFactory(), _config, durationSeconds,
                                                 listener, std::move(recordingFinishedCallback));

    _detector.reset();
}

bool IssueReportingPitchDetector::isRecording() const {
    return _recorder != nullptr;
}

int IssueReportingPitchDetector::realtimePercentage() const {
    return _realtimePercentage.load(std::memory_order_relaxed);
}
}  // namespace saint
