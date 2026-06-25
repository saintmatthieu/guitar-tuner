#pragma once

#include <memory>
#include <vector>

#include "ButterworthFilter.h"
#include "PitchDetectorTypes.h"

namespace saint {
class Preprocessor {
   public:
    // `decimationFactor` D downsamples the low-passed signal by keeping every D-th frame
    // (D=1 is no decimation). The order-6 / 5 kHz Butterworth low-pass doubles as the
    // anti-alias filter; it is comfortably below fs/(2D) for D<=3 and only marginal at D=4.
    // The downstream frequency-domain stages then run at fs/D with a ~D-times smaller FFT.
    Preprocessor(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
                 int decimationFactor);

    void processBlock(float* audio);
    // Returns a reference to an internal buffer, valid until the next call. No
    // per-call heap allocation (real-time-audio path). When decimationFactor > 1 the
    // returned buffer holds the decimated frames, so its size varies slightly from call
    // to call when samplesPerBlockPerChannel is not a multiple of the factor.
    const std::vector<float>& processBlock(const float* audio);

   private:
    // Compact `_scratch` in place, keeping every D-th frame, and resize it to the
    // decimated length. The decimation phase carries across blocks.
    void decimate();

    static constexpr auto cutoffFreq = 5000.0;
    static constexpr auto filterOrder = 6;

    const int _samplesPerBlockPerChannel;
    const int _numChannels;
    const int _decimationFactor;
    int _decimationPhase = 0;

    ButterworthFilter<filterOrder> _lowpass;
    const std::unique_ptr<ButterworthFilter<filterOrder>> _rightLowpass;
    std::vector<float> _scratch;  // reused output buffer
};
}  // namespace saint
