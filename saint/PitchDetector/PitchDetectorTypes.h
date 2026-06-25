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
constexpr auto autocorrAveragingFrameCount = 4;

// Octaviation gate operating point (tuned in eval/gate-tuning-log.md to minimise the
// false-negative rate while keeping the median and 99th-percentile RMS error at or
// under the no-gate reference). octaviationPresenceThreshold is the cut on the fitted
// probNotOctaviated for an unconstrained (fresh) detection; octaviationHarmonicityFloor
// rejects estimates whose octave-corrected fundamental lacks harmonic support, which is
// what lets the presence cut be this permissive without admitting octave errors.
constexpr double octaviationPresenceThreshold = 0.55;
constexpr float octaviationHarmonicityFloor = 0.30f;

// The probNotOctaviated cut applied once an estimate constraint is locked (phase 2 /
// tracking). In the locked phase the autocorrelation search and the octave
// disambiguation are already restricted to within a major third of the constraint, so
// octave errors are essentially impossible; this cut then acts purely as a
// presence/confidence gate, deciding how long the note keeps being tracked as it decays
// into the noise. Lowering it below octaviationPresenceThreshold trades a little RMS for
// a markedly lower FNR (the harmonicity floor still rejects frames that have decayed into
// noise). Lowered from the historical 0.7 to 0.3: recovers ~15% of the weighted FNR
// (0.247 -> 0.210) for an inaudible median-RMS change (2.43 -> 2.60 cents), while keeping
// the catastrophic tail (p99 RMS) and FPR contained. See eval/gate-tuning-log.md for the
// full sweep.
constexpr double octaviationPresenceThresholdWithConstraint = 0.5;

// Octaviation-gate configuration for PitchDetectorImpl. Defaults are the tuned production
// operating point (the constants above). `apply` is a calibration toggle: set it false to
// bypass the probNotOctaviated gate so every frame emits its estimate (used to collect the
// full presence/error distribution for re-fitting the gate; see
// eval/fitAndShowErrorProbabilityModels.py).
struct OctaviationGateConfig {
    bool apply = true;
    double presenceThreshold = octaviationPresenceThreshold;
    float harmonicityFloor = octaviationHarmonicityFloor;
    double presenceThresholdWithConstraint = octaviationPresenceThresholdWithConstraint;
};

// Onset detector (OnsetDetector.h) decision: a level-adaptive threshold on the
// spectral-flux novelty function. An onset fires when the flux exceeds
// onsetFluxMedianMultiplier times a causal running median of the recent flux, floored
// by onsetFluxAbsFloor. Spectral flux scales with input amplitude, so a fixed absolute
// threshold only works near the level it was tuned for (a quiet/unplugged instrument
// then needs hard plucks); dividing by the recent-flux median makes the decision
// input-level invariant while keeping the flux's noise separation. onsetFluxAbsFloor is
// the silence guard: it sets the lowest input level at which onsets are still detected
// (must clear ambient/digital silence yet stay below the quietest target note's running
// median). onsetFluxMedianMultiplier is derived for zero false negatives on the
// calibration corpus (eval/showOnsetDetectionHistograms.py); both are validated against
// OnsetDetectorCalibrationTests and PitchDetectorImplTests.
constexpr float onsetFluxMedianMultiplier = 3.0f;
constexpr float onsetFluxAbsFloor = 0.001f;

// Onset-detector configuration. Defaults are the tuned operating point (the constants above).
struct OnsetDetectorConfig {
    float k = onsetFluxMedianMultiplier;  // multiplier on the running-median flux baseline
    float absFloor = onsetFluxAbsFloor;   // floor on the baseline; guards true silence
};

constexpr auto majorThirdRatio = 1.26f;

// PitchDetectorMedianFilter operating point (production defaults). defaultMedianFilterDuration
// is the median-filter window (s) and drives output latency. The hold keeps emitting the last
// locked pitch through a brief presence dip so the indicator does not blink off; it engages
// only defaultHoldOnsetGuard seconds after an onset, so it acts on a note's settled tail rather
// than its still-resolving attack. See PitchDetectorMedianFilter.h for the full rationale.
constexpr float defaultMedianFilterDuration = 0.15f;
constexpr float defaultHoldDuration = 1.0f;
constexpr float defaultHoldOnsetGuard = 0.5f;

// Preprocessor decimation factor: the frequency-domain stages run at fs/D. D=1 is no
// decimation; D=2 halves the FFT for ~1/3 less process() CPU at no measurable accuracy cost.
constexpr int defaultDecimationFactor = 2;

// Median-filter configuration for PitchDetectorMedianFilter. Defaults are the production
// operating point (the constants above).
struct MedianFilterConfig {
    float filterDuration = defaultMedianFilterDuration;
    float holdDuration = defaultHoldDuration;
    float holdOnsetGuard = defaultHoldOnsetGuard;
};

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
    // Spans from the lowest pitch of any tuning to the highest.
    // Used when the tuning is not known in advance.
    Unknown,
};
}  // namespace saint