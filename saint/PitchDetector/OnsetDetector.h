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
// better than broadband-energy flux, which is dominated by the ringing
// baseline (see OnsetDetectorCalibrationTests).
class OnsetDetector {
   public:
    OnsetDetector(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
                  float minFreq, float threshold = onsetSpectralFluxThreshold);

    bool process(const float* audio, DebugOutput* = nullptr);
    bool process(float* audio, DebugOutput* = nullptr);

   private:
    const ChannelFormat _channelFormat;
    const int _blockSize;
    const float _threshold;
    const std::vector<double> _window;
    const int _fftSize;
    RealFft _fft;
    std::vector<float> _audioBuffer;
    // Magnitude spectrum of the previous frame; empty until the first frame is
    // analysed (so the first flux is 0, like np.diff with prepend).
    std::vector<float> _prevMagnitude;
    const int _leastBlockCountBetweenOffsets;
    int _countSinceLastTrueOutput = 0;
};

}  // namespace saint
