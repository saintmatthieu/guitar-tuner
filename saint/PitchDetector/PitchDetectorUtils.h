#pragma once

#include "PitchDetectorTypes.h"

namespace saint {
// Lower bound of the pitch-search range: the tuning's lowest open-string note shifted by
// `semitoneOffset` semitones (negative = below). The default (-3) leaves room to tune a
// slightly flat string up into range; the benchmark sweeps it (see minFreqSemitoneOffset).
float getMinFreq(Tuning tuning, int semitoneOffset = defaultMinFreqSemitoneOffset);
float getMaxFreq(Tuning tuning);
}  // namespace saint