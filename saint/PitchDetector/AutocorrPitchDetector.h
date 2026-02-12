#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "Cepstrum.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetector.h"
#include "RealFft.h"
#include "Utils.h"

class PitchDetectorLoggerInterface;

namespace saint {
class AutocorrPitchDetector {
   public:
    AutocorrPitchDetector(int sampleRate, int fftSize, const std::vector<float>& fftWindow,
                          float minFreq, PitchDetectorLoggerInterface& logger);

    float process(const std::vector<std::complex<float>>& dft, float* presenceScore,
                  std::optional<float> constraint = std::nullopt);

   private:
    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    RealFft _fwdFft;
    const std::vector<float> _lpWindow;
    const int _lastSearchIndex;
    const std::vector<float> _windowXcor;
};
}  // namespace saint
