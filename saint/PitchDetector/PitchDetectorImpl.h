#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "Cepstrum.h"
#include "PitchDetector.h"
#include "RealFft.h"
#include "Utils.h"

class PitchDetectorLoggerInterface;

namespace saint {
class PitchDetectorImpl : public PitchDetector {
   public:
    PitchDetectorImpl(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
                      const std::optional<Config>& config,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);
    ~PitchDetectorImpl() override = default;

    float process(const float*, float* presenceScore) override;
    int delaySamples() const override {
        return windowSizeSamples() / 2;
    }

    int windowSizeSamples() const {
        return static_cast<int>(_window.size());
    }

   private:
    float refineEstimateBasedOnStrongestHarmonic(const std::vector<float>& logSpectrum,
                                                 float targetFreq) const;

    float disambiguateEstimate(float priorEstimate, const std::vector<float>& dbSpectrum) const;

    float getCepstrumPeakFrequency(const CepstrumData& cepstrumData) const;

    const int _sampleRate;
    const ChannelFormat _channelFormat;
    const int _blockSize;
    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    const utils::WindowType _windowType;
    const std::vector<float> _window;
    const int _fftSize;
    const float _binFreq;
    RealFft _fwdFft;
    CepstrumData _cepstrumData;
    const std::vector<float> _lpWindow;
    const std::vector<float> _windowXcor;
    const int _latencySamples;
    std::vector<float> _audioBuffer;
    const float _minFreq;
    const float _maxFreq;
    const int _lastSearchIndex;
    bool _bufferErrorLoggedAlready = false;
    const std::function<float(float)> _xcorrTransform = [this, i = 0](float x) mutable {
        return x / std::max(_windowXcor[i++], 1e-6f);
    };
};
}  // namespace saint
