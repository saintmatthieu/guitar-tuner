#include "PitchDetectorMedianFilter.h"

#include <algorithm>
#include <cmath>  // ceil

namespace saint {

namespace {
constexpr int getFilterSize(int sampleRate, int blockSize) {
    constexpr auto filterDuration = 0.5f;
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
      _buffer(getFilterSize(sampleRate, blockSize), 0.f),
      _delayedScores((_buffer.size() - 1) / 2, 0.f) {}

float PitchDetectorMedianFilter::process(const float* input, float* presenceScore) {
    return process(input, presenceScore, nullptr);
}

int PitchDetectorMedianFilter::delaySamples() const {
    return _delayedScores.size() * _blockSize + _impl->delaySamples();
}

float PitchDetectorMedianFilter::process(const float* input, float* presenceScore,
                                         float* unfilteredEstimate) {
    _buffer.erase(_buffer.begin());
    float rawPresenceScore = 0.f;
    const auto raw = _impl->process(input, &rawPresenceScore);

    if (presenceScore != nullptr) {
        _delayedScores.push_back(rawPresenceScore);
        *presenceScore = _delayedScores.front();
        _delayedScores.erase(_delayedScores.begin());
    }

    if (unfilteredEstimate != nullptr) {
        *unfilteredEstimate = raw;
    }
    _buffer.push_back(raw);
    auto sortedBuffer = _buffer;
    std::sort(sortedBuffer.begin(), sortedBuffer.end());
    const auto medianFiltered = sortedBuffer[sortedBuffer.size() / 2];

    // Onset/offset detection based on presence score.
    // Unlock only when presence score drops below threshold (note offset).
    constexpr auto unlockThreshold = 0.7f;
    if (_locked && rawPresenceScore < unlockThreshold) {
        _locked = false;
        _impl->setEstimateConstraint(std::nullopt);
    }

    // Lock when median filter outputs a non-zero estimate (note onset).
    // Update the constraint while locked to track the current pitch.
    if (medianFiltered > 0.f) {
        _locked = true;
        _impl->setEstimateConstraint(medianFiltered);
    }

    return medianFiltered;
}

}  // namespace saint