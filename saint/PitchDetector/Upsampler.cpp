#include "Upsampler.h"

namespace saint {
Upsampler::Upsampler(int sampleRate)
    : _filter(1, 0,
              designFIRLowpass<numTaps>(std::min(sampleRate * 0.45, 8000.), sampleRate * factor,
                                        WindowType::Blackman)) {}

std::vector<float> Upsampler::process(const float* audio, int numSamples) {
    std::vector<float> output(numSamples * factor, 0.f);
    for (auto i = 0; i < numSamples; ++i) {
        output[i * factor] = audio[i];
    }
    _filter.process(output.data(), numSamples * factor);
    return output;
}
}  // namespace saint