#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "Cepstrum.h"
#include "PitchDetectorTypes.h"
#include "RealFft.h"

class PitchDetectorLoggerInterface;

namespace saint {
class AutocorrEstimateDisambiguator {
   public:
    AutocorrEstimateDisambiguator(int sampleRate, int fftSize,
                                  const std::optional<PitchDetectorConfig>& config,
                                  PitchDetectorLoggerInterface& logger);

    float process(float xcorrEstimate, const std::vector<std::complex<float>>& spectrum);

   private:
    float refineEstimateBasedOnStrongestHarmonic(const std::vector<float>& logSpectrum,
                                                 float targetFreq) const;

    float disambiguateEstimate(float priorEstimate, const std::vector<float>& idealSpectrum) const;

    float getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const;

    void toIdealSpectrum(std::vector<float>& logSpectrum);

    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    const float _binFreq;
    RealFft _cepstrumFft;
    const float _minFreq;
    const float _maxFreq;
};
}  // namespace saint
