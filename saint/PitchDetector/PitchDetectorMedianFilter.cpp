#include "PitchDetectorMedianFilter.h"

#include <algorithm>
#include <cmath>    // ceil
#include <numeric>  // accumulate

namespace saint {

namespace {
int getFilterSize(int sampleRate, int blockSize, float filterDuration) {
    const auto blockDuration = static_cast<float>(blockSize) / static_cast<float>(sampleRate);
    auto size = static_cast<int>(std::ceil(filterDuration / blockDuration));
    if (size % 2 == 0) {
        ++size;  // Make it odd, it's simpler for median calculation
    }
    return std::max(size, 3);  // minimum 3 for median to be meaningful
}

int durationToBlocks(int sampleRate, int blockSize, float duration) {
    const auto blockDuration = static_cast<float>(blockSize) / static_cast<float>(sampleRate);
    return std::max(0, static_cast<int>(std::round(duration / blockDuration)));
}
}  // namespace

PitchDetectorMedianFilter::PitchDetectorMedianFilter(int sampleRate, int blockSize,
                                                     std::unique_ptr<PitchDetectorImpl> impl,
                                                     float filterDuration, float holdDuration,
                                                     float holdOnsetGuard)
    : _blockSize(blockSize),
      _impl(std::move(impl)),
      _buffer(getFilterSize(sampleRate, blockSize, filterDuration)),
      _delayedScores((_buffer.size() - 1) / 2, 0.f),
      _maxHoldFrames(durationToBlocks(sampleRate, blockSize, holdDuration)),
      _minFramesBeforeHold(durationToBlocks(sampleRate, blockSize, holdOnsetGuard)) {}

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
        // New attack: drop the lock and any held pitch from the previous note, and restart
        // the post-onset counter that guards the hold.
        _allGoodOnce = false;
        _heldPitch = 0.f;
        _framesHeld = 0;
        _framesSinceOnset = 0;
    } else {
        ++_framesSinceOnset;
    }

    const auto rawPresenceScore = debugOutput->at("presenceScore");
    _delayedScores.push_back(rawPresenceScore);
    (*debugOutput)["presenceScore"] = _delayedScores.front();
    _delayedScores.erase(_delayedScores.begin());

    _buffer.push_back(raw);
    if (!_allGoodOnce) {
        const auto allNonZero =
            std::all_of(_buffer.begin(), _buffer.end(), [](float raw) { return raw > 0.f; });
        if (allNonZero) {
            const auto minEstimate = *std::min_element(_buffer.begin(), _buffer.end());
            const auto maxEstimate = *std::max_element(_buffer.begin(), _buffer.end());
            _allGoodOnce = maxEstimate / minEstimate < majorThirdRatio;
        }
    }

    if (!_allGoodOnce) {
        return 0.f;
    }

    auto sortedBuffer = _buffer;
    std::sort(sortedBuffer.begin(), sortedBuffer.end());
    const auto medianFiltered = sortedBuffer[sortedBuffer.size() / 2];

    if (medianFiltered > 0.f) {
        // Tracking: update the constraint to follow the current pitch, remember it as the
        // value to hold, and re-arm the hold window.
        _impl->setEstimateConstraint(medianFiltered);
        _heldPitch = medianFiltered;
        _framesHeld = 0;
        return medianFiltered;
    }

    // Locked but no estimate this block: the presence score dipped under the gate. Hold the
    // last pitch for up to _maxHoldFrames so the indicator doesn't blink off on a transient
    // dip, but only once we are well past the onset (_framesSinceOnset >= _minFramesBeforeHold)
    // so we never hold a still-settling attack. The constraint is left in place, so tracking
    // resumes from the same anchor the moment the note re-emerges (medianFiltered > 0 above).
    // Past the cap the held note is considered gone and we emit 0; the lock is kept (it is
    // dropped only on an onset), so a later re-emergence still resumes and re-arms the hold.
    if (_heldPitch > 0.f && _framesHeld < _maxHoldFrames &&
        _framesSinceOnset >= _minFramesBeforeHold) {
        ++_framesHeld;
        return _heldPitch;
    }

    return 0.f;
}

}  // namespace saint