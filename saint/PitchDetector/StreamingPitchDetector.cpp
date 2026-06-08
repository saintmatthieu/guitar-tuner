#include "StreamingPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace saint {

StreamingPitchDetector::StreamingPitchDetector(std::unique_ptr<PitchDetector> inner, int blockSize)
    : _inner(std::move(inner)), _blockSize(std::max(1, blockSize)) {
    assert(_inner);
}

size_t StreamingPitchDetector::push(const float* samples, size_t n) {
    if (!samples || n == 0) {
        return 0;
    }
    _buffer.insert(_buffer.end(), samples, samples + n);
    return n;
}

bool StreamingPitchDetector::processOnce(float& hzOut, DebugOutput* debugOutput) {
    hzOut = 0.f;

    if (_buffer.size() - _readPos < static_cast<size_t>(_blockSize)) {
        // Not enough for a full block. Drop the already-consumed prefix so the buffer tracks only
        // the unconsumed tail (< blockSize) rather than growing without bound. This is the single
        // compaction point: it moves at most blockSize samples, once per drain cycle, instead of
        // shifting the whole buffer on every consumed block.
        if (_readPos > 0) {
            _buffer.erase(_buffer.begin(), _buffer.begin() + _readPos);
            _readPos = 0;
        }
        return false;
    }

    const float hz = _inner->process(_buffer.data() + _readPos, debugOutput, nullptr);
    _readPos += _blockSize;

    hzOut = (hz > 0.f && std::isfinite(hz)) ? hz : 0.f;
    return true;
}

void StreamingPitchDetector::clear() noexcept {
    _buffer.clear();
    _readPos = 0;
}

}  // namespace saint
