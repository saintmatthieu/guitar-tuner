#include "PitchDetectorMedianFilter.h"

#include <algorithm>
#include <cmath>  // ceil

#include "PitchDetectorUtils.h"

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
}  // namespace

PitchDetectorMedianFilter::PitchDetectorMedianFilter(int sampleRate, int blockSize,
                                                     std::unique_ptr<InRangePitchDetector> impl,
                                                     MedianFilterConfig config)
    : _blockSize(blockSize),
      _impl(std::move(impl)),
      _buffer(getFilterSize(sampleRate, blockSize, config.filterDuration)),
      _delayedScores((_buffer.size() - 1) / 2, 0.f) {}

int PitchDetectorMedianFilter::delaySamples() const {
    return _delayedScores.size() * _blockSize + _impl->delaySamples();
}

std::pair<float, float> PitchDetectorMedianFilter::pitchSearchRange() const {
    return _impl->pitchSearchRange();
}

PitchDetectionResult PitchDetectorMedianFilter::process(const float* input,
                                                        DebugOutput* debugOutput,
                                                        std::vector<float>* debugOutputSignal) {
    _buffer.erase(_buffer.begin());

    if (debugOutput == nullptr) {
        debugOutput = &_debugOutput;
    }
    debugOutput->clear();

    const auto raw = _impl->process(input, debugOutput, debugOutputSignal);
    if (debugOutput->at("isOnset") == 1.f) {
        // New attack: drop the median-filter lock so it re-converges on the new note. The
        // downstream PitchDetectionHolder bridges the resulting output gap, so a fresh note's
        // attack does not blink the indicator off.
        _allGoodOnce = false;
    }
    if (raw.bucket.has_value() && _lastBucket.has_value() && *raw.bucket != *_lastBucket) {
        // The verdict moved to a different bucket - the detector has decided that what it was
        // reporting as an in-range pitch is a harmonic of a string sounding below the range, or
        // the other way round. The two readings are an octave or more apart, so keeping the
        // window would only fail its own consistency check for as long as both live in it.
        // Re-lock instead and let the new bucket's readings fill it; downstream the hold bridges
        // the gap. (Rescaling the window by the two readings' ratio instead - they are, after
        // all, two readings of one periodicity - was measured: it buys 0.002 of weighted FNR and
        // costs twice the bucket error and 4 cents of 99th-percentile RMS.)
        _allGoodOnce = false;
        std::fill(_buffer.begin(), _buffer.end(), 0.f);
    }
    if (raw.bucket.has_value()) {
        _lastBucket = raw.bucket;
    }

    const auto rawPresenceScore = debugOutput->at("presenceScore");
    _delayedScores.push_back(rawPresenceScore);
    (*debugOutput)["presenceScore"] = _delayedScores.front();
    _delayedScores.erase(_delayedScores.begin());

    _buffer.push_back(raw.pitch);
    if (!_allGoodOnce) {
        const auto allNonZero =
            std::all_of(_buffer.begin(), _buffer.end(), [](float pitch) { return pitch > 0.f; });
        if (allNonZero) {
            const auto minEstimate = *std::min_element(_buffer.begin(), _buffer.end());
            const auto maxEstimate = *std::max_element(_buffer.begin(), _buffer.end());
            _allGoodOnce = maxEstimate / minEstimate < majorThirdRatio;
        }
    }

    if (!_allGoodOnce) {
        return {};
    }

    auto sortedBuffer = _buffer;
    std::sort(sortedBuffer.begin(), sortedBuffer.end());
    const auto medianFiltered = sortedBuffer[sortedBuffer.size() / 2];

    if (medianFiltered > 0.f) {
        const auto bucket = getBucket(medianFiltered, pitchSearchRange());
        if (bucket == PitchBucket::inRange) {
            // Tracking: update the constraint to follow the current pitch. A below-range
            // pitch must not become the constraint: the in-range search would then be
            // clamped to a major third around a frequency it cannot even reach, and the
            // detector would go silent until the next onset cleared the lock.
            _impl->setEstimateConstraint(medianFiltered);
        }
        return {medianFiltered, bucket};
    }

    return {};
}

}  // namespace saint