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

// --- Noise-compensated autocorrelation -------------------------------------
// The autocorrelation is the inverse FFT of the (low-pass-weighted) power
// spectrum |X|^2. Under additive, uncorrelated background noise the measured
// power spectrum is E|X|^2 = |S|^2 + |N|^2, so subtracting an estimate of the
// noise power spectrum |N|^2 before the inverse FFT removes the noise's
// contribution to the autocorrelation. This restores the prominence of the
// periodicity peak (and hence the presence score) in noisy conditions, which
// is where the detector is most prone to false negatives.
//
// |N[k]|^2 is tracked per frequency bin by exponential averaging, frozen while
// a pitch is present (presence score above noiseUpdatePresenceThreshold) so
// that sustained notes never leak into the noise estimate.

// Over-subtraction factor (alpha): multiple of the estimated noise power to
// subtract. >1 trades a little signal distortion for more aggressive removal.
constexpr float noiseOverSubtractionFactor = 1.0f;
// Spectral floor (beta): residual power is never allowed below this fraction of
// the original, bounding the gain and preventing the normalising zero-lag value
// from collapsing on noise-only frames.
constexpr float noiseSpectralFloor = 0.1f;
// Per-frame smoothing for the recursive noise-power estimate.
constexpr float noiseEstimateSmoothing = 0.95f;
// Frames whose (normalised) presence score is below this are treated as
// noise-only and update the noise estimate; above it the estimate is frozen.
// Chosen well below the detection gate (0.7/0.85) so only confidently aperiodic
// frames contribute.
constexpr float noiseUpdatePresenceThreshold = 0.5f;
// Number of noise-only frames to accumulate before subtraction engages.
constexpr int noiseEstimateWarmupFrames = 5;

struct Pitch {
    const PitchClass pitchClass;
    const int octave;
};

enum class ChannelFormat { Mono = 1, Stereo = 2 };
constexpr int numChannels(ChannelFormat format) {
    return static_cast<int>(format);
}

struct PitchDetectorConfig {
    const std::optional<Pitch> lowestPitch;
    const std::optional<Pitch> highestPitch;
};
}  // namespace saint