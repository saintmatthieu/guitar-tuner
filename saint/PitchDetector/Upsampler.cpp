#include "Upsampler.h"

namespace saint {
Upsampler::Upsampler(int sampleRate, int factor, int cutoffFreqHz)
    : _filter(1, 0,
              designFIRLowpass<numTaps>(std::min<double>(sampleRate * 0.45, cutoffFreqHz),
                                        sampleRate * factor, WindowType::Blackman)),
      _factor(factor) {}

std::vector<float> Upsampler::process(const float* audio, int numSamples) {
    std::vector<float> output(numSamples * _factor, 0.f);
    for (auto i = 0; i < numSamples; ++i) {
        output[i * _factor] = audio[i];
    }
    if (_factor > 1)
        _filter.process(output.data(), numSamples * _factor);
    return output;
}
}  // namespace saint