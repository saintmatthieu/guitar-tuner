#include "AutocorrPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "Autocorrelation.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
std::vector<float> getLpWindow(int sampleRate, int fftSize) {
    std::vector<float> window(fftSize / 2);
    const int cutoffBin = std::min(fftSize / 2, fftSize * autocorrCutoffFreqHz / sampleRate);
    const int rollOffSize = fftSize * autocorrRolloffHz / sampleRate;
    std::fill(window.begin(), window.begin() + cutoffBin, 1.f);
    for (auto i = 0; i < rollOffSize && cutoffBin + rollOffSize < fftSize / 2; ++i) {
        window[cutoffBin + i] = 1.f - i / static_cast<float>(rollOffSize);
    }
    std::fill(window.begin() + cutoffBin + rollOffSize, window.end(), 0.f);
    return window;
}

}  // namespace

AutocorrPitchDetector::AutocorrPitchDetector(int sampleRate, int fftSize,
                                             const std::vector<float>& fftWindow, float minFreq,
                                             PitchDetectorLoggerInterface& logger)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _fwdFft(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / minFreq))),
      _windowXcorr(getWindowXCorr(_fwdFft, fftWindow, _lpWindow)),
      _xcorr(static_cast<size_t>(_fftSize), 0.f),
      _freqScratch(static_cast<size_t>(_fftSize) / 2) {
    if (autocorrAveragingFrameCount > 1) {
        _xcorrHistory.assign(autocorrAveragingFrameCount, std::vector<float>(_fftSize, 0.f));
        _averagedXcorr.resize(_fftSize, 0.f);
    }
}

void AutocorrPitchDetector::reset() {
    _historyWritePos = 0;
    _historyFilled = 0;
}

const std::vector<float>& AutocorrPitchDetector::averageOverFrames(
    const std::vector<float>& xcorr) {
    if (autocorrAveragingFrameCount <= 1) {
        return xcorr;
    }

    // Overwrite the oldest slot (equal-size copy, no reallocation in the RT path).
    _xcorrHistory[_historyWritePos] = xcorr;
    _historyWritePos = (_historyWritePos + 1) % autocorrAveragingFrameCount;
    if (_historyFilled < autocorrAveragingFrameCount) {
        ++_historyFilled;
    }

    // Average the valid frames. While the ring is filling, only [0, _historyFilled)
    // hold real data; once full, _historyFilled == K and every slot is valid.
    std::fill(_averagedXcorr.begin(), _averagedXcorr.end(), 0.f);
    for (auto k = 0; k < _historyFilled; ++k) {
        const auto& frame = _xcorrHistory[k];
        for (auto i = 0u; i < _averagedXcorr.size(); ++i) {
            _averagedXcorr[i] += frame[i];
        }
    }
    const auto scale = 1.f / _historyFilled;
    for (auto& v : _averagedXcorr) {
        v *= scale;
    }
    return _averagedXcorr;
}

float AutocorrPitchDetector::process(const std::vector<std::complex<float>>& freq,
                                     float* presenceScore, std::optional<float> constraint) {
    _logger.Log(_windowXcorr.data(), _windowXcorr.size(), "windowXcorr");

    // Compute cross-correlation. getXCorr overwrites its spectrum argument in place,
    // so work on a reused copy and leave the caller's `freq` untouched (the impl still
    // needs it for the power spectrum).
    std::copy(freq.begin(), freq.end(), _freqScratch.begin());
    getXCorr(_fwdFft, _xcorr, _freqScratch, _lpWindow);
    _logger.Log(_xcorr.data(), _xcorr.size(), "xcorr");

    // Average over the last few frames to suppress noise-driven octave errors
    // (idea #1). `acf` aliases `_xcorr` when averaging is disabled.
    const std::vector<float>& acf = averageOverFrames(_xcorr);
    _logger.Log(acf.data(), acf.size(), "averagedXcorr");

    // Determine search range based on constraint
    int firstSearchIndex = 0;
    int lastSearchIndex = _lastSearchIndex;

    if (constraint.has_value() && constraint.value() > 0.f) {
        const auto constraintFreq = constraint.value();
        const auto minFreq = constraintFreq / majorThirdRatio;
        const auto maxFreq = constraintFreq * majorThirdRatio;
        // Convert frequencies to lag indices (frequency = sampleRate / lag)
        // Higher frequency means smaller lag
        firstSearchIndex = std::max(0, static_cast<int>(_sampleRate / maxFreq));
        lastSearchIndex = std::min(_lastSearchIndex, static_cast<int>(_sampleRate / minFreq) + 1);
    }

    const auto peak = findAutocorrPeak(acf, _windowXcorr, firstSearchIndex, lastSearchIndex);
    if (presenceScore) {
        *presenceScore = peak.presence;
    }

    if (peak.lag == 0) {
        return 0.f;
    }

    const auto fracIndex = utils::quadFit(&acf[peak.lag - 1]);
    return static_cast<float>(_sampleRate) / (peak.lag + fracIndex);
}
}  // namespace saint
