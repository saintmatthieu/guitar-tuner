#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace saint {
using DebugOutput = std::unordered_map<std::string, float>;

enum class PitchClass { C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B, OneKiloHz /*for testing*/ };

static constexpr auto autocorrCutoffFreqHz = 1500;
static constexpr auto autocorrRolloffHz = 200;
// Assuming that we're only dealing with 6-string guitars in standard tuning and standard sample
// rates, the worst case of quantization when estimating the frequency from reading the
// autocorrelation is an E4 at 44.1kHz.
// Upsampling by a factor of 4, we reduce the maximal quantization to 1.5 cents.
constexpr auto autocorrUpsamplingFactor = 4;

constexpr auto majorThirdRatio = 1.26f;

struct Pitch {
    const PitchClass pitchClass;
    const int octave;
};

enum class ChannelFormat { Mono = 1, Stereo = 2 };
constexpr int numChannels(ChannelFormat format) {
    return static_cast<int>(format);
}

enum class Tuning {
    Standard,
    HalfStepDown,
    DTuning,
    CSharpTuning,
    CTuning,
    BTuning,
    BbTuning,
    ATuning,
    DropD,
    DropDb,
    DropC,
    DropB,
    DropBb,
    DropA,
    DropA2,
    DoubleDropD,
    DoubleDropDb,
    DoubleDropC,
    DoubleDropB,
    DoubleDropBb,
    DoubleDropA,
    OpenA,
    OpenA2,
    OpenA3,
    OpenAm,
    OpenAm2,
    OpenCSharp,
    OpenC,
    OpenCsus2,
    OpenC6,
    OpenD,
    OpenDm,
    OpenDsus2,
    OpenDsus4Celtic,
    OpenD5,
    OpenE,
    OpenEm,
    OpenEsus2,
    OpenEsus4,
    OpenE5,
    OpenEm11,
    OpenEb,
    OpenF,
    OpenF2,
    OpenF3,
    OpenFm,
    OpenG,
    OpenG2,
    OpenGm,
    OpenGsus2,
    OpenGsus4,
    OpenGsus42,
    PerfectFourthTuning,
    NewStandardTuning,
};
}  // namespace saint