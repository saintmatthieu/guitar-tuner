#include "PitchDetectorMedianFilter.h"

#include <algorithm>
#include <cmath>    // ceil
#include <numeric>  // accumulate

namespace saint {

namespace {
constexpr int getFilterSize(int sampleRate, int blockSize) {
    constexpr auto filterDuration = 0.2f;
    const auto blockDuration = static_cast<float>(blockSize) / static_cast<float>(sampleRate);
    auto size = static_cast<int>(std::ceil(filterDuration / blockDuration));
    if (size % 2 == 0) {
        ++size;  // Make it odd, it's simpler for median calculation
    }
    return size;
}
}  // namespace

PitchDetectorMedianFilter::PitchDetectorMedianFilter(int sampleRate, int blockSize,
                                                     std::unique_ptr<PitchDetectorImpl> impl)
    : _blockSize(blockSize),
      _impl(std::move(impl)),
      _buffer(getFilterSize(sampleRate, blockSize)),
      _delayedScores((_buffer.size() - 1) / 2, 0.f) {}

int PitchDetectorMedianFilter::delaySamples() const {
    return _delayedScores.size() * _blockSize + _impl->delaySamples();
}

float PitchDetectorMedianFilter::process(const float* input, DebugOutput* debugOutput,
                                         std::vector<float>* debugOutputSignal) {
    _buffer.erase(_buffer.begin());

    if (debugOutput == nullptr) {
        debugOutput = &_debugOutput;
    }

    const auto raw = _impl->process(input, debugOutput, debugOutputSignal);
    if (const auto isOnset = debugOutput->at("isOnset") == 1.f) {
        _allGoodOnce = false;
    }

    const auto rawPresenceScore = debugOutput->at("presenceScore");
    _delayedScores.push_back(rawPresenceScore);
    (*debugOutput)["presenceScore"] = _delayedScores.front();
    _delayedScores.erase(_delayedScores.begin());

    _buffer.push_back(raw);
    _allGoodOnce |=
        std::all_of(_buffer.begin(), _buffer.end(), [](float raw) { return raw > 0.f; });

    if (!_allGoodOnce) {
        return 0.f;
    }

    auto sortedBuffer = _buffer;
    std::sort(sortedBuffer.begin(), sortedBuffer.end());
    const auto medianFiltered = sortedBuffer[sortedBuffer.size() / 2];

    // Lock when median filter outputs a non-zero estimate (note onset).
    // Update the constraint while locked to track the current pitch.
    if (medianFiltered > 0.f) {
        _impl->setEstimateConstraint(medianFiltered);
    }

    return medianFiltered;
}

}  // namespace saint