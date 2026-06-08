#include "ReblockingPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace saint {

ReblockingPitchDetector::ReblockingPitchDetector(std::unique_ptr<FixedBlockPitchDetector> inner,
                                                 int blockSize)
    : _inner(std::move(inner)), _blockSize(std::max(1, blockSize)) {
    assert(_inner);
}

float ReblockingPitchDetector::process(const float* input, int n, DebugOutput* debugOutput,
                                       std::vector<float>* debugOutputSignal) {
    if (input != nullptr && n > 0) {
        _buffer.insert(_buffer.end(), input, input + n);
    }

    float hz = 0.f;
    bool processedAnyBlock = false;
    while (_buffer.size() - _readPos >= static_cast<size_t>(_blockSize)) {
        hz = _inner->process(_buffer.data() + _readPos, debugOutput, debugOutputSignal);
        _readPos += _blockSize;
        processedAnyBlock = true;
    }

    // Drop the consumed prefix, keeping only the unconsumed tail (< blockSize). Single compaction
    // point: moves at most blockSize samples per call, rather than shifting on every block.
    if (_readPos > 0) {
        _buffer.erase(_buffer.begin(), _buffer.begin() + _readPos);
        _readPos = 0;
    }

    if (!processedAnyBlock) {
        return 0.f;
    }
    return (hz > 0.f && std::isfinite(hz)) ? hz : 0.f;
}

int ReblockingPitchDetector::delaySamples() const {
    return _inner->delaySamples();
}

}  // namespace saint
