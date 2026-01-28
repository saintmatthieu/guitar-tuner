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
    : _innerDetector(std::move(innerDetector)),
      _buffer(getFilterSize(sampleRate, blockSize), 0.f) {}

float PitchDetectorMedianFilter::process(const float* input, float* presenceScore) {
    //
    _buffer.erase(_buffer.begin());
    _buffer.push_back(_innerDetector->process(input, presenceScore));
    auto sortedBuffer = _buffer;
    std::sort(sortedBuffer.begin(), sortedBuffer.end());
    return sortedBuffer[sortedBuffer.size() / 2];
}
}  // namespace saint