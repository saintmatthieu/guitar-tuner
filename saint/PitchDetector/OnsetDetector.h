#pragma once

#include <complex>
#include <vector>

#include "PitchDetectorTypes.h"
#include "RealFft.h"

namespace saint {

// Detects note onsets from a spectral-flux novelty function: the sum over
// frequency bins of the positive change in magnitude between consecutive
// frames. A pluck injects energy across the note's whole harmonic series at
// once (large positive flux), whereas a decaying/ringing note has flat-or-
// falling magnitudes (~zero flux) and steady noise has small frame-to-frame
// changes. This separates a re-pluck from the still-ringing previous note far
// better than broadband-energy flux (see OnsetDetectorCalibrationTests).
//
// The decision is level-adaptive: an onset fires when the flux exceeds `k` times
// a causal running median of the recent flux, floored by `absFloor` to guard
// silence. Spectral flux scales with input amplitude, so an absolute threshold
// only works at the level it was tuned for; dividing by the recent-flux median
// makes the decision invariant to input level (a quiet/unplugged instrument
// triggers without plucking hard) while preserving the flux's noise separation.
class OnsetDetector {
   public:
    OnsetDetector(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
                  float minFreq, OnsetDetectorConfig = {});

    bool process(const float* audio, DebugOutput* = nullptr);
    bool process(float* audio, DebugOutput* = nullptr);

   private:
    const ChannelFormat _channelFormat;
    const int _blockSize;
    const float _k;         // multiplier on the running-median flux baseline
    const float _absFloor;  // floor on the baseline; guards true silence
    const std::vector<double> _window;
    const int _fftSize;
    RealFft _fft;
    std::vector<float> _audioBuffer;
    // Reused scratch (windowed time block and its spectrum), so process() allocates
    // nothing on the audio thread.
    std::vector<float> _timeScratch;
    std::vector<std::complex<float>> _freqScratch;
    // Magnitude spectrum of the previous frame; empty until the first frame is
    // analysed (so the first flux is 0, like np.diff with prepend).
    std::vector<float> _prevMagnitude;
    // Causal ring buffer of recent flux values for the adaptive-threshold median,
    // pre-seeded with _absFloor so the first attack still triggers during warm-up.
    std::vector<float> _fluxHistory;
    std::vector<float> _medianScratch;  // reused buffer for the per-frame median
    int _fluxHistoryPos = 0;
    const int _leastBlockCountBetweenOffsets;
    int _countSinceLastTrueOutput = 0;
};

}  // namespace saint
