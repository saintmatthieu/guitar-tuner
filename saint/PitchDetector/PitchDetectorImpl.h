#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "AutocorrPitchDetector.h"
#include "Cepstrum.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetector.h"
#include "RealFft.h"
#include "Utils.h"

class PitchDetectorLoggerInterface;

namespace saint {
class PitchDetectorImpl : public PitchDetector {
   public:
    PitchDetectorImpl(FrequencyDomainTransformer, AutocorrPitchDetector, int sampleRate,
                      const std::optional<PitchDetectorConfig>& config,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);
    ~PitchDetectorImpl() override = default;

    float process(const float*, float* presenceScore) override;
    int delaySamples() const override {
        return windowSizeSamples() / 2;
    }

    int windowSizeSamples() const {
        return _frequencyDomainTransformer.windowSizeSamples();
    }

   private:
    float refineEstimateBasedOnStrongestHarmonic(const std::vector<float>& logSpectrum,
                                                 float targetFreq) const;

    float disambiguateEstimate(float priorEstimate, const std::vector<float>& idealSpectrum) const;

    float getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const;

    void toIdealSpectrum(std::vector<float>& logSpectrum);

    FrequencyDomainTransformer _frequencyDomainTransformer;
    AutocorrPitchDetector _autocorrPitchDetector;

    const int _sampleRate;
    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    const int _fftSize;
    const float _binFreq;
    RealFft _cepstrumFft;
    const float _minFreq;
    const float _maxFreq;
};
}  // namespace saint
