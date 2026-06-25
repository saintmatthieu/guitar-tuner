#pragma once

#include <memory>
#include <vector>

#include "PitchDetector.h"
#include "PitchDetectorImpl.h"

namespace saint {
class PitchDetectorMedianFilter : public PitchDetector {
   public:
    PitchDetectorMedianFilter(int sampleRate, int blockSize,
                              std::unique_ptr<PitchDetectorImpl> impl, MedianFilterConfig = {});

    ~PitchDetectorMedianFilter() override = default;

    float process(const float* input, DebugOutput*, std::vector<float>*) override;
    int delaySamples() const override;

   private:
    float getHeldPitch();

    const int _blockSize = 0;
    const std::unique_ptr<PitchDetectorImpl> _impl;
    DebugOutput _debugOutput;
    std::vector<float> _buffer;
    std::vector<float> _delayedScores;
    bool _allGoodOnce = false;

    // "Hold" state: once locked, a brief presence dip drops the impl's estimate to 0,
    // which would blink the pitch indicator off. Instead, keep emitting the last pitch
    // for up to _maxHoldFrames blocks (the constraint is left in place, so tracking
    // resumes from the same anchor the moment the note re-emerges). Kept short — it only
    // smooths over blinks, not real silence; an app wanting longer persistence does so
    // itself. _maxHoldFrames == 0 disables the hold (legacy behaviour).
    //
    // The hold only engages once at least _minFramesBeforeHold blocks have elapsed since
    // the last onset: it is meant for the settled tail of a note, not the attack, where
    // the anchor is still being established and holding a half-resolved estimate would
    // inflate the error tail.
    const int _maxHoldFrames;
    float _heldPitch = 0.f;
    int _framesHeld = 0;
    int _framesSinceOnset = 0;
};
}  // namespace saint