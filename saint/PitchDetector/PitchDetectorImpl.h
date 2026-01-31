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
using Spectrum = Aligned<std::vector<std::complex<float>>>;

class PitchDetectorImpl : public PitchDetector {
   public:
    PitchDetectorImpl(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
                      const std::optional<Config>& config,
                      std::unique_ptr<PitchDetectorLoggerInterface> logger);
    float process(const float*, float* presenceScore) override;

    int windowSizeSamples() {
        return static_cast<int>(_window.size());
    }

   private:
    float getXCorrNormalizedMaximum(const std::vector<float>& xcorr) const;

    const float _sampleRate;
    const ChannelFormat _channelFormat;
    const int _blockSize;
    const std::unique_ptr<PitchDetectorLoggerInterface> _logger;
    const utils::WindowType _windowType;
    const std::vector<float> _window;
    const int _fftSize;
    RealFft _fwdFft;
    CepstrumData _cepstrumData;
    const std::vector<float> _lpWindow;
    const float _minFreq;
    const float _maxFreq;
    const int _lastSearchIndex;
    const std::vector<float> _windowXcor;
    const float _coefs;
    Spectrum _filteredSpectrum;

    const int _latencySamples;
    std::vector<float> _audioBuffer;
    bool _bufferErrorLoggedAlready = false;
    const std::function<float(float)> _xcorrTransform = [this, i = 0](float x) mutable {
        return x / std::max(_windowXcor[i++], 1e-6f);
    };
};
}  // namespace saint
