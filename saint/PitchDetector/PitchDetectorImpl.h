#pragma once

#include <memory>
#include <optional>

#include "AutocorrEstimateDisambiguator.h"
#include "AutocorrPitchDetector.h"
#include "FrequencyDomainTransformer.h"
#include "OnsetDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
class PitchDetectorImpl {
   public:
    PitchDetectorImpl(int sampleRate, int windowSize, int fftSize, FrequencyDomainTransformer,
                      AutocorrPitchDetector, AutocorrEstimateDisambiguator, OnsetDetector,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);

    float process(const float*, DebugOutput*, std::vector<float>* debugOutputSignal = nullptr);
    int delaySamples() const {
        return windowSizeSamples() / 2;
    }

    void setEstimateConstraint(float constraint) {
        _estimateConstraint = constraint;
    }
    void clearEstimateConstraint() {
        _estimateConstraint.reset();
    }

    int windowSizeSamples() const {
        return _frequencyDomainTransformer.windowSizeSamples();
    }

   private:
    void toIdealSpectrum(std::vector<float>& logSpectrum);

    const int _sampleRate;
    const int _fftSize;
    const float _binFreq;
    const double _binResolution;  // if the fft size is larger than the window size

    FrequencyDomainTransformer _frequencyDomainTransformer;
    AutocorrPitchDetector _autocorrPitchDetector;
    AutocorrEstimateDisambiguator _disambiguator;
    OnsetDetector _onsetDetector;

    RealFft _cepstrumFft;
    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    std::optional<float> _estimateConstraint;
};
}  // namespace saint
