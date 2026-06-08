#include "AutocorrPitchDetector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "AutocorrPitchDetector.h"
#include "PitchDetectorLoggerInterface.h"

namespace saint {
namespace {
void getXCorr(RealFft& fft, std::vector<float>& time, std::vector<std::complex<float>> freq,
              const std::vector<float>& lpWindow) {
    auto timeData = time.data();

    for (auto i = 0u; i < lpWindow.size(); ++i) {
        auto& X = freq[i];
        X *= lpWindow[i] * std::complex<float>{X.real(), -X.imag()};
    }
    std::fill(freq.data() + lpWindow.size(), freq.data() + fft.size / 2, 0.f);
    fft.inverse(freq.data(), timeData);
    if (timeData[0] < 1e-6f) {
        return;
    }
    const auto normalizer = 1.f / timeData[0];
    for (auto i = 0; i < fft.size; ++i) {
        timeData[i] *= normalizer;
    }
}

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

std::vector<std::complex<float>> getSpectrum(RealFft& fft, const float* timeData) {
    Aligned<std::vector<std::complex<float>>> freqAligned;
    auto& freq = freqAligned.value;
    freq.resize(fft.size / 2);
    fft.forward(timeData, freq.data());
    return freqAligned.value;
}

// The low-pass window is a run of positive weights followed by zeros, so the
// bins that actually contribute to the autocorrelation are a prefix. Those are
// the only bins worth tracking and subtracting noise from.
int countPositiveBins(const std::vector<float>& lpWindow) {
    int count = 0;
    for (const auto w : lpWindow) {
        if (w > 0.f)
            ++count;
    }
    return count;
}

std::vector<float> getWindowXCorr(RealFft& fft, const std::vector<float>& window,
                                  const std::vector<float>& lpWindow) {
    Aligned<std::vector<float>> xcorrAligned;
    auto& xcorr = xcorrAligned.value;
    xcorr.resize(fft.size);
    std::copy(window.begin(), window.end(), xcorr.begin());
    std::fill(xcorr.begin() + window.size(), xcorr.end(), 0.f);
    std::vector<std::complex<float>> freq = getSpectrum(fft, xcorr.data());
    getXCorr(fft, xcorr, freq, lpWindow);
    return xcorr;
}
}  // namespace

AutocorrPitchDetector::AutocorrPitchDetector(int sampleRate, int fftSize,
                                             const std::vector<float>& fftWindow, float minFreq,
                                             PitchDetectorLoggerInterface& logger,
                                             bool noiseCompensation)
    : _sampleRate(sampleRate),
      _logger(logger),
      _fftSize(fftSize),
      _fwdFft(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _lastSearchIndex(std::min(_fftSize / 2, static_cast<int>(sampleRate / minFreq))),
      _windowXcorr(getWindowXCorr(_fwdFft, fftWindow, _lpWindow)),
      _noiseCompensation(noiseCompensation),
      _noiseBinCount(countPositiveBins(_lpWindow)),
      _noisePsd(_noiseBinCount, 0.f) {}

float AutocorrPitchDetector::process(const std::vector<std::complex<float>>& freq,
                                     float* presenceScore, std::optional<float> constraint) {
    std::vector<float> xcorr(_fftSize);
    _logger.Log(_windowXcorr.data(), _windowXcorr.size(), "windowXcorr");

    // Noise-compensated autocorrelation: subtract the estimated noise power
    // spectrum from |X|^2 before computing the autocorrelation. The subtraction
    // is realised as a real-valued spectral gain applied to the complex
    // spectrum, so getXCorr (which forms |X|^2 internally) sees the denoised
    // power. We only subtract once a note is active (the estimate has been
    // frozen at the onset) and a noise floor has actually been learned from the
    // pre-roll; otherwise the spectrum is left untouched and the detector
    // behaves exactly as baseline (including the noise-only pre-roll, so the
    // gate's false-positive behaviour there is unchanged).
    std::vector<std::complex<float>> compensated;
    const std::vector<std::complex<float>>* xcorrInput = &freq;
    if (_noiseCompensation && _noiseEstimationFrozen &&
        _noiseFramesSeen >= noiseEstimateWarmupFrames) {
        compensated = freq;
        applyNoiseCompensation(compensated);
        xcorrInput = &compensated;
    }

    // Compute cross-correlation
    getXCorr(_fwdFft, xcorr, *xcorrInput, _lpWindow);
    _logger.Log(xcorr.data(), xcorr.size(), "xcorr");

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

    auto maxIndex = 0;
    auto wentNegative = false;
    auto maximum = 0.f;
    for (auto i = 0; i < lastSearchIndex; ++i) {
        wentNegative |= xcorr[i] < 0;
        if (wentNegative && i >= firstSearchIndex && xcorr[i] > maximum) {
            maximum = xcorr[i];
            maxIndex = i;
        }
    }

    maximum /= _windowXcorr[maxIndex];
    if (presenceScore) {
        *presenceScore = maximum;
    }

    // Update the noise-power estimate from this frame's raw spectrum, gated by
    // the presence score so frames that contain a pitch don't pollute it.
    updateNoiseEstimate(freq, maximum);

    if (maxIndex == 0) {
        return 0.f;
    }

    const auto fracIndex = utils::quadFit(&xcorr[maxIndex - 1]);
    const auto refinedIndex = maxIndex + fracIndex;

    return maxIndex == 0 ? 0.f : static_cast<float>(_sampleRate) / refinedIndex;
}

void AutocorrPitchDetector::applyNoiseCompensation(
    std::vector<std::complex<float>>& spectrum) const {
    for (int i = 0; i < _noiseBinCount; ++i) {
        const float power = std::norm(spectrum[i]);  // |X[i]|^2
        if (power <= 1e-20f) {
            continue;
        }
        const float denoised = std::max(power - noiseOverSubtractionFactor * _noisePsd[i],
                                        noiseSpectralFloor * power);
        // Scaling the complex bin by sqrt(denoised / power) leaves its phase
        // untouched while making |scaled|^2 == denoised.
        spectrum[i] *= std::sqrt(denoised / power);
    }
}

void AutocorrPitchDetector::updateNoiseEstimate(const std::vector<std::complex<float>>& spectrum,
                                                float presenceScore) {
    if (!_noiseCompensation || _noiseEstimationFrozen) {
        return;
    }
    // Secondary guard: even before the first onset, skip any frame that looks
    // periodic (e.g. an onset detected a frame late), so a note edge can't seed
    // the noise floor.
    if (presenceScore >= noiseUpdatePresenceThreshold) {
        return;
    }
    // The first observed noise-only frame seeds the estimate outright; later
    // frames are smoothed in.
    const float smoothing = _noiseFramesSeen == 0 ? 0.f : noiseEstimateSmoothing;
    for (int i = 0; i < _noiseBinCount; ++i) {
        const float power = std::norm(spectrum[i]);
        _noisePsd[i] = smoothing * _noisePsd[i] + (1.f - smoothing) * power;
    }
    ++_noiseFramesSeen;
}
}  // namespace saint
