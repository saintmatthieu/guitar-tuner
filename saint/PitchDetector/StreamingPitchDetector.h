#pragma once

#include <memory>
#include <vector>

#include "FixedBlockPitchDetector.h"
#include "PitchDetectorTypes.h"

namespace saint {
// Adapts an arbitrary-sized input stream to the fixed-block PitchDetector contract.
//
// PitchDetector::process() requires exactly `blockSize` samples per call (see
// FixedBlockPitchDetector.h), and is fed non-overlapping blocks back-to-back. Real-time audio
// callbacks do not honour that on their own: iOS AVAudioEngine taps treat the requested buffer size
// as a hint, Android AudioRecord::read() may return short reads, and Web Audio render quanta (128
// frames) rarely equal `blockSize`. This class accumulates whatever is pushed and releases it to
// the inner detector one `blockSize` block at a time, in arrival order, with no overlap or gaps —
// so the inner detector sees exactly the same block sequence it would from an ideal fixed-size
// source.
class StreamingPitchDetector {
   public:
    StreamingPitchDetector(std::unique_ptr<FixedBlockPitchDetector> inner, int blockSize);

    // Append `n` mono samples to the internal buffer. Returns the number accepted (`n`).
    size_t push(const float* samples, size_t n);

    // If at least one full block is buffered, run the inner detector on the next block, write its
    // pitch estimate (0 if none) to `hzOut`, optionally populate `debugOutput`, and return true.
    // Otherwise leave `hzOut` at 0 and return false.
    bool processOnce(float& hzOut, DebugOutput* debugOutput = nullptr);

    // Discard buffered audio. Does not reset the inner detector's internal state.
    void clear() noexcept;

    int blockSize() const noexcept {
        return _blockSize;
    }

   private:
    const std::unique_ptr<FixedBlockPitchDetector> _inner;
    const int _blockSize;
    std::vector<float> _buffer;
    size_t _readPos = 0;  // index of the first unconsumed sample in _buffer
};
}  // namespace saint
