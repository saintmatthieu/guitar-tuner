#include "RecordingPitchDetector.h"

#include <algorithm>
#include <cassert>

namespace saint {
RecordingPitchDetector::RecordingPitchDetector(std::unique_ptr<PitchDetector> inner,
                                               recording::PitchDetectorConfig config,
                                               int durationSeconds, IRecordingListener& listener,
                                               OnComplete onComplete)
    : _config(config),
      _inner(std::move(inner)),
      _listener(listener),
      _onComplete(std::move(onComplete)),
      _samplesPerBlock(config.samplesPerBlockPerChannel * numChannels(config.channelFormat)),
      _maxNumBlocks(
          std::max(1, durationSeconds * config.sampleRate / config.samplesPerBlockPerChannel)) {
    assert(_inner);
    _buffer.resize(static_cast<size_t>(_maxNumBlocks) * _samplesPerBlock);
}

float RecordingPitchDetector::process(const float* input, DebugOutput* debugOutput,
                                      std::vector<float>* debugOutputSignal) {
    assert(_inner);  // must not be called after the recording completed
    if (!_inner) {
        return 0.f;
    }
    std::copy(input, input + _samplesPerBlock,
              _buffer.begin() + static_cast<size_t>(_numBlocksStored) * _samplesPerBlock);
    ++_numBlocksStored;
    const auto remainingSamples =
        (_maxNumBlocks - _numBlocksStored) * _config.samplesPerBlockPerChannel;
    const auto remainingSeconds =
        (remainingSamples + _config.sampleRate - 1) / _config.sampleRate;  // rounded up
    if (remainingSeconds != _lastReportedRemainingSeconds) {
        _lastReportedRemainingSeconds = remainingSeconds;
        _listener.onProgress(remainingSeconds);
    }
    const auto result = _inner->process(input, debugOutput, debugOutputSignal);
    if (_numBlocksStored == _maxNumBlocks) {
        complete();
    }
    return result;
}

int RecordingPitchDetector::delaySamples() const {
    return _inner ? _inner->delaySamples() : 0;
}

void RecordingPitchDetector::stop() {
    complete();
}

void RecordingPitchDetector::complete() {
    if (!_inner) {
        return;
    }
    // Shrinks - doesn't allocate.
    _buffer.resize(static_cast<size_t>(_numBlocksStored) * _samplesPerBlock);
    if (_onComplete) {
        _onComplete(std::move(_inner), recording::RecordingData{_config, std::move(_buffer)});
    }
}
}  // namespace saint
