#pragma once

#include <vector>

#include "PitchDetectorTypes.h"

namespace saint {
// Evidence that a partial really sits at `index` in a floor-subtracted spectrum (levels at or
// below the noise floor being <= 0): how far it rises above the two points `halfSpacing` bins
// either side of it, but no further than it rises above the floor. Both halves are needed. A
// level alone means nothing where neighbouring partials are not resolved, because their skirts
// fill the positions in between; a local rise alone means nothing either, because the noise
// floor wiggles by a few dB everywhere, and summed over a comb that wiggle would pass for a
// harmonic series.
float spectralProminence(const std::vector<float>& spectrum, float index, float halfSpacing);

// Share of a comb's prominence that sits off the explained sub-comb, measured *per comb member*
// rather than in total. Which members are explained depends on the ratio between the two
// frequencies - an estimate an octave above the fundamental explains every second harmonic, one
// an octave and a fifth above only every third - so totals make the same physical evidence score
// differently for different ratios (with every harmonic equally present, 1/2 against 2/3), and no
// single threshold fits both. Per member, a comb whose harmonics are all equally present scores
// 0.5 whatever the ratio. Returns 0 when either side has no member to average, there being
// nothing to compare then.
float combSupport(float explainedSum, int explainedCount, float offCombSum, int offCombCount);

// Lower bound of the pitch-search range: the tuning's lowest open-string note shifted by
// `semitoneOffset` semitones (negative = below). The default (-3) leaves room to tune a
// slightly flat string up into range; the benchmark sweeps it (see minFreqSemitoneOffset).
float getMinFreq(Tuning tuning, int semitoneOffset = defaultMinFreqSemitoneOffset);
float getMaxFreq(Tuning tuning);
}  // namespace saint