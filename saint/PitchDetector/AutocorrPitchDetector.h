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

    // Drop the cross-frame averaging history. Call on onset so a new note's
    // autocorrelation never blurs into the previous note's.
    void reset();

   private:
    // Push the latest autocorrelation frame into the ring buffer and return the
    // average over the last autocorrAveragingFrameCount frames (see idea #1 in
    // acf-denoising-ideas.md). Returns xcorr unchanged when averaging is disabled.
    const std::vector<float>& averageOverFrames(const std::vector<float>& xcorr);

    const int _sampleRate;
    PitchDetectorLoggerInterface& _logger;
    const int _fftSize;
    RealFft _fwdFft;
    const std::vector<float> _lpWindow;
    const int _lastSearchIndex;
    const std::vector<float> _windowXcorr;

    // Cross-frame averaging state (idea #1). Empty/unused when averaging is off.
    std::vector<std::vector<float>> _xcorrHistory;
    std::vector<float> _averagedXcorr;
    int _historyWritePos = 0;
    int _historyFilled = 0;
};
}  // namespace saint
