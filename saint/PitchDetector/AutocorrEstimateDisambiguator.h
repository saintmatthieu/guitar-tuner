#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "Cepstrum.h"
#include "PitchDetectorTypes.h"
#include "RealFft.h"
#include "Utils/CommonTypes.h"

class PitchDetectorLoggerInterface;

namespace saint {
class AutocorrEstimateDisambiguator {
   public:
    AutocorrEstimateDisambiguator(int sampleRate, int fftSize, Tuning tuning,
                                  PitchDetectorLoggerInterface& logger);

    float process(float priorEstimate, const std::vector<PeakModel>& spectrumModel,
                  std::optional<float> constraint = std::nullopt) const;

   private:
    float disambiguateEstimate(float priorEstimate, const std::vector<PeakModel>& idealSpectrum,
                               std::optional<float> constraint) const;
    float getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const;

    PitchDetectorLoggerInterface& _logger;
    const float _binFreq;
    const float _minFreq;
    const float _maxFreq;
};
}  // namespace saint
