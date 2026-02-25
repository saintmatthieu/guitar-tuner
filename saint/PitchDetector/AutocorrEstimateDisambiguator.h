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

    void updateNoiseProfile(float confidence, const std::vector<float>& dbSpectrum);
    float disambiguate(float xcorrEstimate, const std::vector<float>& dbSpectrum,
                       const std::optional<float>& constraint = std::nullopt) const;

   private:
    void toIdealSpectrum(std::vector<float>& logSpectrum) const;
    std::vector<float> getSpectrumEnvelope(const std::vector<float>& dbSpectrum);

    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    const float _binFreq;
    RealFft _cepstrumFft;
    const float _minFreq;
    const float _maxFreq;
    std::vector<float> _noiseProfile;
    const float _threshold = 0.88f;
    const float _maxAlpha = 0.03f;
};
}  // namespace saint
