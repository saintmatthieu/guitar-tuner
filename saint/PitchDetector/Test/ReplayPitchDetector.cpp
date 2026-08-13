#include "ReplayPitchDetector.h"

#include "PitchDetectorFactory.h"
#include "RecordingFileReader.h"
// createInstance() returns a unique_ptr<IssueReportingPitchDetector>; the complete type is
// needed to convert it to the unique_ptr<PitchDetector> stored in `_inner`.
#include "Recording/IssueReportingPitchDetector.h"

namespace saint {
std::unique_ptr<ReplayPitchDetector> ReplayPitchDetector::fromFile(
    const std::filesystem::path& path, std::string* warning) {
    auto data = recording::readWavFile(path, warning);
    if (!data.has_value()) {
        return nullptr;
    }
    return std::unique_ptr<ReplayPitchDetector>(new ReplayPitchDetector(std::move(*data)));
}

ReplayPitchDetector::ReplayPitchDetector(recording::RecordingData data)
    : _data(std::move(data)),
      _inner(PitchDetectorFactory::createInstance(
          _data.config.sampleRate, _data.config.channelFormat,
          _data.config.samplesPerBlockPerChannel, _data.config.tuning)),
      _samplesPerBlock(_data.config.samplesPerBlockPerChannel *
                       numChannels(_data.config.channelFormat)) {}

PitchDetectionResult ReplayPitchDetector::process(const float*, DebugOutput* debugOutput,
                                                  std::vector<float>* debugOutputSignal) {
    if (_blockIndex >= numBlocks()) {
        return {};
    }
    const auto* block =
        _data.interleaved.data() + static_cast<size_t>(_blockIndex) * _samplesPerBlock;
    ++_blockIndex;
    return _inner->process(block, debugOutput, debugOutputSignal);
}

int ReplayPitchDetector::delaySamples() const {
    return _inner->delaySamples();
}

std::pair<float, float> ReplayPitchDetector::pitchSearchRange() const {
    return _inner->pitchSearchRange();
}

const recording::PitchDetectorConfig& ReplayPitchDetector::config() const {
    return _data.config;
}

int ReplayPitchDetector::numBlocks() const {
    return static_cast<int>(_data.interleaved.size()) / _samplesPerBlock;
}

int ReplayPitchDetector::numBlocksLeft() const {
    return numBlocks() - _blockIndex;
}

const float* ReplayPitchDetector::peekBlock() const {
    if (_blockIndex >= numBlocks()) {
        return nullptr;
    }
    return _data.interleaved.data() + static_cast<size_t>(_blockIndex) * _samplesPerBlock;
}
}  // namespace saint
