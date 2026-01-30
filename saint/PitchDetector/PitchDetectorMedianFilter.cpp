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
                                                     std::unique_ptr<PitchDetector> innerDetector)
    : _blockSize(blockSize),
      _innerDetector(std::move(innerDetector)),
      _buffer(getFilterSize(sampleRate, blockSize), 0.f),
      _delayedScores((_buffer.size() - 1) / 2, 0.f) {}

float PitchDetectorMedianFilter::process(const float* input, float* presenceScore) {
    return process(input, presenceScore, nullptr);
}

int PitchDetectorMedianFilter::delaySamples() const {
    return _delayedScores.size() * _blockSize + _innerDetector->delaySamples();
}

float PitchDetectorMedianFilter::process(const float* input, float* presenceScore,
                                         float* unfilteredEstimate) {
    _buffer.erase(_buffer.begin());
    const auto raw = _innerDetector->process(input, presenceScore);

    if (presenceScore != nullptr) {
        _delayedScores.push_back(*presenceScore);
        *presenceScore = _delayedScores.front();
        _delayedScores.erase(_delayedScores.begin());
    }

    if (unfilteredEstimate != nullptr) {
        *unfilteredEstimate = raw;
    }
    _buffer.push_back(raw);
    auto sortedBuffer = _buffer;
    std::sort(sortedBuffer.begin(), sortedBuffer.end());
    return sortedBuffer[sortedBuffer.size() / 2];
}

}  // namespace saint