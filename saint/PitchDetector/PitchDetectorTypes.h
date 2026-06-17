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

// Number of consecutive autocorrelation frames to average before peak picking.
// The ACF is shift-invariant, so a sustained note's signal peak adds coherently
// across frames while random noise averages down (variance ~1/K), which curbs
// the noise-driven octave-jump errors. The average is reset on onset so a new
// note never blurs into the previous one. 1 disables averaging (legacy behaviour).
constexpr auto autocorrAveragingFrameCount = 1;

// Octaviation gate operating point (tuned in eval/gate-tuning-log.md to minimise the
// false-negative rate while keeping the median and 99th-percentile RMS error at or
// under the no-gate reference). octaviationPresenceThreshold is the cut on the fitted
// probNotOctaviated for an unconstrained (fresh) detection; octaviationHarmonicityFloor
// rejects estimates whose octave-corrected fundamental lacks harmonic support, which is
// what lets the presence cut be this permissive without admitting octave errors.
constexpr double octaviationPresenceThreshold = 0.20;
constexpr float octaviationHarmonicityFloor = 0.20f;

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