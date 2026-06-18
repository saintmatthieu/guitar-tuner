#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "Cepstrum.h"
#include "PitchDetectorTypes.h"
#include "RealFft.h"

class PitchDetectorLoggerInterface;

namespace saint {
class AutocorrEstimateDisambiguator {
   public:
    AutocorrEstimateDisambiguator(int sampleRate, int fftSize, Tuning tuning,
                                  PitchDetectorLoggerInterface& logger);

    // If harmonicityOut is non-null, it receives a [0,1] harmonicity score for the
    // returned estimate: the fraction of whitened-spectrum peak energy that lies on
    // the higher-harmonic (k>=2) comb of the estimate. Low for pure tones, broadband
    // noise and inharmonic/octave-misplaced locks; high for genuine harmonic notes.
    float process(float xcorrEstimate, const std::vector<float>& dbSpectrum,
                  std::optional<float> constraint = std::nullopt, float* harmonicityOut = nullptr);

   private:
    float disambiguateEstimate(float priorEstimate, const std::vector<float>& idealSpectrum,
                               std::optional<float> constraint) const;
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
