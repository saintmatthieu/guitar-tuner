#include "PitchDetectorUtils.h"

#include <cmath>
#include <unordered_map>

namespace {
constexpr float defaultLeastFrequencyToDetect = /* A0 */ 27.5f;

float pitchToFrequency(const saint::Pitch& pitch) {
    using namespace saint;
    if (PitchClass::OneKiloHz == pitch.pitchClass) {
        return 1000 * (1 << pitch.octave);
    }
    const std::unordered_map<PitchClass, int> semitonesFromA{
        {PitchClass::C, -9},  {PitchClass::Db, -8}, {PitchClass::D, -7},  {PitchClass::Eb, -6},
        {PitchClass::E, -5},  {PitchClass::F, -4},  {PitchClass::Gb, -3}, {PitchClass::G, -2},
        {PitchClass::Ab, -1}, {PitchClass::A, 0},   {PitchClass::Bb, 1},  {PitchClass::B, 2},
    };
    const int semitonesFromA4 = semitonesFromA.at(pitch.pitchClass) + (pitch.octave - 4) * 12;
    return 440.f * std::pow(2.f, semitonesFromA4 / 12.f);
}
}  // namespace

float saint::getMinFreq(const std::optional<PitchDetectorConfig>& config) {
    return config && config->lowestPitch.has_value() ? pitchToFrequency(*config->lowestPitch)
                                                     : defaultLeastFrequencyToDetect;
}

float saint::getMaxFreq(const std::optional<PitchDetectorConfig>& config) {
    return config && config->highestPitch.has_value() ? pitchToFrequency(*config->highestPitch)
                                                      : 2000.f;
}