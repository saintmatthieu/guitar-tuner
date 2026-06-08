#pragma once

#include <memory>
#include <vector>

#include "FixedBlockPitchDetector.h"
#include "PitchDetector.h"

namespace saint {
// Variable-input-size PitchDetector built on top of a fixed-block one.
//
// A FixedBlockPitchDetector requires exactly `blockSize` samples per call, fed as non-overlapping
// blocks back-to-back. Real-time audio callbacks rarely honour that: iOS AVAudioEngine taps treat
// the requested buffer size as a hint, Android AudioRecord::read() may return short reads, and Web
// Audio render quanta (128 frames) seldom equal `blockSize`. This class buffers whatever is pushed
// and releases it to the inner detector one `blockSize` block at a time, in arrival order, with no
// overlap or gaps — so the inner detector sees exactly the block sequence it would from an ideal
// fixed-size source.
class ReblockingPitchDetector : public PitchDetector {
   public:
    ReblockingPitchDetector(std::unique_ptr<FixedBlockPitchDetector> inner, int blockSize);

    float process(const float* input, int n, DebugOutput* debugOutput = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const std::unique_ptr<FixedBlockPitchDetector> _inner;
    const int _blockSize;
    std::vector<float> _buffer;
    size_t _readPos = 0;  // index of the first unconsumed sample in _buffer
};
}  // namespace saint
