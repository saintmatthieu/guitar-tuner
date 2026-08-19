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
                                                     std::unique_ptr<OutRangePitchDetector> inner,
                                                     MedianFilterConfig config)
    : _blockSize(blockSize),
      _inner(std::move(inner)),
      _buffer(getFilterSize(sampleRate, blockSize, config.filterDuration)),
      _delayedScores((_buffer.size() - 1) / 2, 0.f) {}

int PitchDetectorMedianFilter::delaySamples() const {
    return _delayedScores.size() * _blockSize + _inner->delaySamples();
}

std::pair<float, float> PitchDetectorMedianFilter::pitchSearchRange() const {
    return _inner->pitchSearchRange();
}

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

    if (raw.bucket.has_value()) {
        _lastBucket = raw.bucket;
    }

    const auto rawPresenceScore = debugOutput->at("presenceScore");
    _delayedScores.push_back(rawPresenceScore);
    (*debugOutput)["presenceScore"] = _delayedScores.front();
    _delayedScores.erase(_delayedScores.begin());

    _buffer.push_back(raw);
    if (!_allGoodOnce) {
        const auto allSamePitchedBucket =
            _lastBucket.has_value() &&
            std::all_of(_buffer.begin(), _buffer.end(),
                        [this](const PitchDetectionResult& r) { return r.bucket == _lastBucket; });

        if (allSamePitchedBucket && _lastBucket != PitchBucket::inRange) {
            // All either too-low or too-high estimates - that's consistent.
            _allGoodOnce = true;
        } else {
            // In-range estimates. See if are within a major third.

            const auto minEstimate =
                std::min_element(_buffer.begin(), _buffer.end(),
                                 [](const PitchDetectionResult& a, const PitchDetectionResult& b) {
                                     return a.pitch < b.pitch;
                                 })
                    ->pitch;

            const auto maxEstimate =
                std::max_element(_buffer.begin(), _buffer.end(),
                                 [](const PitchDetectionResult& a, const PitchDetectionResult& b) {
                                     return a.pitch < b.pitch;
                                 })
                    ->pitch;

            _allGoodOnce = maxEstimate / minEstimate < majorThirdRatio;
        }
    }

    if (!_allGoodOnce) {
        return {};
    }

    std::optional<PitchBucket> mostRepresentedBucket;
    {
        std::unordered_map<int, int> bucketCounts;
        for (const auto& r : _buffer) {
            const auto key = r.bucket.has_value() ? static_cast<int>(*r.bucket) : -1;
            ++bucketCounts[key];
        }
        const auto index =
            std::max_element(bucketCounts.begin(), bucketCounts.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; })
                ->first;
        if (index >= 0) {
            mostRepresentedBucket = static_cast<PitchBucket>(index);
        }
    }

    if (mostRepresentedBucket != PitchBucket::inRange) {
        return {0.f, mostRepresentedBucket};
    }

    std::vector<float> pitches;
    for (const auto& r : _buffer) {
        if (r.bucket == PitchBucket::inRange) {
            pitches.push_back(r.pitch);
        }
    }

    std::sort(pitches.begin(), pitches.end());
    const auto medianFiltered = pitches[pitches.size() / 2];
    _inner->setEstimateConstraint(medianFiltered);

    return {medianFiltered, PitchBucket::inRange};
}

}  // namespace saint