#include "PitchDetectorUtils.h"

#include <cassert>
#include <cmath>
#include <unordered_map>

namespace {
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

float saint::getMinFreq(Tuning tuning) {
    switch (tuning) {
        // E2 lowest → E2 - 3 = Db2
        case Tuning::Standard:
        case Tuning::OpenA:
        case Tuning::OpenA2:
        case Tuning::OpenA3:
        case Tuning::OpenAm:
        case Tuning::OpenAm2:
        case Tuning::OpenE:
        case Tuning::OpenEm:
        case Tuning::OpenEsus2:
        case Tuning::OpenEsus4:
        case Tuning::OpenE5:
        case Tuning::OpenEm11:
        case Tuning::PerfectFourthTuning:
            return pitchToFrequency({saint::PitchClass::Db, 2});
        // Eb2 lowest → Eb2 - 3 = C2
        case Tuning::HalfStepDown:
        case Tuning::OpenEb:
            return pitchToFrequency({saint::PitchClass::C, 2});
        // D2 lowest → D2 - 3 = B1
        case Tuning::DTuning:
        case Tuning::DropD:
        case Tuning::DoubleDropD:
        case Tuning::OpenD:
        case Tuning::OpenDm:
        case Tuning::OpenDsus2:
        case Tuning::OpenDsus4Celtic:
        case Tuning::OpenD5:
        case Tuning::OpenG:
        case Tuning::OpenGm:
        case Tuning::OpenGsus2:
        case Tuning::OpenGsus4:
        case Tuning::OpenGsus42:
            return pitchToFrequency({saint::PitchClass::B, 1});
        // Db2 lowest → Db2 - 3 = Bb1
        case Tuning::CSharpTuning:
        case Tuning::DropDb:
        case Tuning::DoubleDropDb:
        case Tuning::OpenCSharp:
            return pitchToFrequency({saint::PitchClass::Bb, 1});
        // C2 lowest → C2 - 3 = A1
        case Tuning::CTuning:
        case Tuning::DropC:
        case Tuning::DoubleDropC:
        case Tuning::OpenC:
        case Tuning::OpenCsus2:
        case Tuning::OpenC6:
        case Tuning::OpenF2:
        case Tuning::NewStandardTuning:
            return pitchToFrequency({saint::PitchClass::A, 1});
        // B1 lowest → B1 - 3 = Ab1
        case Tuning::BTuning:
        case Tuning::DropB:
        case Tuning::DoubleDropB:
            return pitchToFrequency({saint::PitchClass::Ab, 1});
        // Bb1 lowest → Bb1 - 3 = G1
        case Tuning::BbTuning:
        case Tuning::DropBb:
        case Tuning::DoubleDropBb:
            return pitchToFrequency({saint::PitchClass::G, 1});
        // A1 lowest → A1 - 3 = Gb1
        case Tuning::ATuning:
        case Tuning::DropA:
        case Tuning::DropA2:
        case Tuning::DoubleDropA:
            return pitchToFrequency({saint::PitchClass::Gb, 1});
        // G2 lowest → G2 - 3 = E2
        case Tuning::OpenG2:
            return pitchToFrequency({saint::PitchClass::E, 2});
        // F2 lowest → F2 - 3 = D2
        case Tuning::OpenF:
        case Tuning::OpenF3:
        case Tuning::OpenFm:
            return pitchToFrequency({saint::PitchClass::D, 2});
        default:
            assert(false);
            return getMinFreq(Tuning::Standard);
    }
}

float saint::getMaxFreq(Tuning tuning) {
    switch (tuning) {
        // E4 highest → E4 + 3 = G4
        case Tuning::Standard:
        case Tuning::DropD:
        case Tuning::DropA2:
        case Tuning::OpenA:
        case Tuning::OpenA2:
        case Tuning::OpenAm:
        case Tuning::OpenAm2:
        case Tuning::OpenC:
        case Tuning::OpenC6:
        case Tuning::OpenE:
        case Tuning::OpenEm:
        case Tuning::OpenEsus2:
        case Tuning::OpenEsus4:
        case Tuning::OpenE5:
            return pitchToFrequency({saint::PitchClass::G, 4});
        // Eb4 highest → Eb4 + 3 = Gb4
        case Tuning::HalfStepDown:
        case Tuning::DropDb:
        case Tuning::OpenEb:
            return pitchToFrequency({saint::PitchClass::Gb, 4});
        // D4 highest → D4 + 3 = F4
        case Tuning::DTuning:
        case Tuning::DropC:
        case Tuning::DoubleDropD:
        case Tuning::OpenD:
        case Tuning::OpenDm:
        case Tuning::OpenDsus2:
        case Tuning::OpenDsus4Celtic:
        case Tuning::OpenD5:
        case Tuning::OpenG:
        case Tuning::OpenG2:
        case Tuning::OpenGm:
        case Tuning::OpenGsus2:
        case Tuning::OpenGsus4:
        case Tuning::OpenGsus42:
        case Tuning::OpenCsus2:
        case Tuning::OpenEm11:
            return pitchToFrequency({saint::PitchClass::F, 4});
        // Db4 highest → Db4 + 3 = E4
        case Tuning::CSharpTuning:
        case Tuning::DropB:
        case Tuning::DoubleDropDb:
        case Tuning::OpenCSharp:
        case Tuning::OpenA3:
            return pitchToFrequency({saint::PitchClass::E, 4});
        // C4 highest → C4 + 3 = Eb4
        case Tuning::CTuning:
        case Tuning::DropBb:
        case Tuning::DoubleDropC:
            return pitchToFrequency({saint::PitchClass::Eb, 4});
        // B3 highest → B3 + 3 = D4
        case Tuning::BTuning:
        case Tuning::DropA:
        case Tuning::DoubleDropB:
            return pitchToFrequency({saint::PitchClass::D, 4});
        // Bb3 highest → Bb3 + 3 = Db4
        case Tuning::BbTuning:
        case Tuning::DoubleDropBb:
            return pitchToFrequency({saint::PitchClass::Db, 4});
        // A3 highest → A3 + 3 = C4
        case Tuning::ATuning:
        case Tuning::DoubleDropA:
            return pitchToFrequency({saint::PitchClass::C, 4});
        // G4 highest → G4 + 3 = Bb4
        case Tuning::NewStandardTuning:
            return pitchToFrequency({saint::PitchClass::Bb, 4});
        // F4 highest → F4 + 3 = Ab4
        case Tuning::OpenF:
        case Tuning::OpenF2:
        case Tuning::OpenF3:
        case Tuning::OpenFm:
        case Tuning::PerfectFourthTuning:
            return pitchToFrequency({saint::PitchClass::Ab, 4});
        default:
            assert(false);
            return getMaxFreq(Tuning::Standard);
    }
}