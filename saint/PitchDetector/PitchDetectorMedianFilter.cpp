#include "PitchDetectorMedianFilter.h"

#include <algorithm>
#include <array>
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
                                                     std::unique_ptr<OutRangePitchDetector> inner,
                                                     MedianFilterConfig config)
    : _blockSize(blockSize),
      _minInRangeCount(config.minInRangeCount),
      _inner(std::move(inner)),
      _buffer(getFilterSize(sampleRate, blockSize, config.filterDuration)),
      _delayedScores((_buffer.size() - 1) / 2, 0.f) {}

int PitchDetectorMedianFilter::delaySamples() const {
    return _delayedScores.size() * _blockSize + _inner->delaySamples();
}

std::pair<float, float> PitchDetectorMedianFilter::pitchSearchRange() const {
    return _inner->pitchSearchRange();
}

namespace {
auto toInt(const std::optional<PitchBucket>& bucket) {
    return bucket.has_value() ? static_cast<int>(bucket.value()) : -1;
}
}  // namespace

PitchDetectionResult PitchDetectorMedianFilter::process(const float* input,
                                                        DebugOutput* debugOutput,
                                                        std::vector<float>* debugOutputSignal) {
    _buffer.erase(_buffer.begin());

    if (debugOutput == nullptr) {
        debugOutput = &_debugOutput;
    }
    debugOutput->clear();

    const auto raw = _inner->process(input, debugOutput, debugOutputSignal);
    if (debugOutput->at("isOnset") == 1.f) {
        // New attack: drop the median-filter lock so it re-converges on the new note. The
        // downstream PitchDetectionHolder bridges the resulting output gap, so a fresh note's
        // attack does not blink the indicator off.
        _allGoodOnce = false;
    }

    const auto rawPresenceScore = debugOutput->at("presenceScore");
    _delayedScores.push_back(rawPresenceScore);
    (*debugOutput)["presenceScore"] = _delayedScores.front();
    _delayedScores.erase(_delayedScores.begin());

    _buffer.push_back({raw.pitch, toInt(raw.bucket)});
    if (!_allGoodOnce) {
        const auto allSamePitchedBucket =
            raw.bucket.has_value() &&
            std::all_of(_buffer.begin(), _buffer.end(),
                        [&raw](const InnerResult& r) { return r.bucket == toInt(raw.bucket); });

        if (allSamePitchedBucket && raw.bucket != PitchBucket::inRange) {
            // All either too-low or too-high estimates - that's consistent.
            _allGoodOnce = true;
        } else {
            // In-range estimates. See if are within a major third.

            const auto minEstimate =
                std::min_element(
                    _buffer.begin(), _buffer.end(),
                    [](const InnerResult& a, const InnerResult& b) { return a.pitch < b.pitch; })
                    ->pitch;

            const auto maxEstimate =
                std::max_element(
                    _buffer.begin(), _buffer.end(),
                    [](const InnerResult& a, const InnerResult& b) { return a.pitch < b.pitch; })
                    ->pitch;

            _allGoodOnce = maxEstimate / minEstimate < majorThirdRatio;
        }
    }

    if (!_allGoodOnce) {
        return {};
    }

    // Counted in place: the bucket is one of four values, so this needs no map.
    std::array<int, 4> bucketCounts{};
    for (const auto& r : _buffer) {
        ++bucketCounts[r.bucket + 1];
    }
    if (std::max_element(bucketCounts.begin(), bucketCounts.end()) == bucketCounts.begin()) {
        // No pitch
        return {};
    }

    auto& sorted = _sortScratch;
    sorted.assign(_buffer.begin(), _buffer.end());
    sorted.erase(std::remove_if(sorted.begin(), sorted.end(),
                                [](const InnerResult& r) { return r.bucket == -1; }),
                 sorted.end());

    // First sort by bucket
    std::sort(sorted.begin(), sorted.end(),
              [](const InnerResult& a, const InnerResult& b) { return a.bucket < b.bucket; });

    const auto medianBucket = sorted[sorted.size() / 2].bucket;
    if (medianBucket != static_cast<int>(PitchBucket::inRange)) {
        return {0.f, static_cast<PitchBucket>(medianBucket)};
    }

    sorted.erase(std::remove_if(sorted.begin(), sorted.end(),
                                [](const InnerResult& r) {
                                    return r.bucket != static_cast<int>(PitchBucket::inRange);
                                }),
                 sorted.end());

    if (static_cast<int>(sorted.size()) < _minInRangeCount) {
        return {};
    }

    // Now sort by pitch
    std::sort(sorted.begin(), sorted.end(),
              [](const InnerResult& a, const InnerResult& b) { return a.pitch < b.pitch; });

    const auto medianFiltered = sorted[sorted.size() / 2].pitch;
    _inner->setEstimateConstraint(medianFiltered);

    return {medianFiltered, PitchBucket::inRange};
}

}  // namespace saint