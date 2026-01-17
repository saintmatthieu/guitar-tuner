#include "PitchDetectorImpl.h"
#include "DummyFormantShifterLogger.h"
#include "FormantShifterLogger.h"
#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <math.h>
#include <numeric>
#include <optional>

namespace saint {

std::unique_ptr<PitchDetector> PitchDetector::createInstance(int sampleRate) {
  const auto debug =
      utils::getEnvironmentVariableAsBool("SAINT_DEBUG_PITCHDETECTOR");
  constexpr auto A1Frequency = 55.0f;
  if (debug && utils::isDebugBuild()) {
    return std::make_unique<PitchDetectorImpl>(
        sampleRate, A1Frequency,
        std::make_unique<FormantShifterLogger>(sampleRate, 0.2 * sampleRate));
  } else {
    return std::make_unique<PitchDetectorImpl>(
        sampleRate, A1Frequency, std::make_unique<DummyFormantShifterLogger>());
  }
}

namespace {
constexpr auto twoPi = 6.283185307179586f;
constexpr auto cutoffFreq = 1500;

constexpr float FastLog2(float x) {
  static_assert(sizeof(float) == sizeof(int32_t));
  union {
    float val;
    int32_t x;
  } u = {x};
  auto log_2 = (float)(((u.x >> 23) & 255) - 128);
  u.x &= ~(255 << 23);
  u.x += 127 << 23;
  log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val - 0.65871759316667f;
  return log_2;
}

constexpr float log2ToDb = 20 / 3.321928094887362f;

int getFftOrder(int windowSize) {
  return static_cast<int>(ceilf(log2f((float)windowSize)));
}

int getFftSizeSamples(int windowSize) { return 1 << getFftOrder(windowSize); }

int getWindowSizeSamples(int sampleRate,
                         const std::optional<float> &leastFrequencyToDetect) {
  // If not provided, use the lower open-E of a guitar.
  const auto freq =
      leastFrequencyToDetect.has_value() ? *leastFrequencyToDetect : 83.f;

  // 3.3 times the fundamental period. More and that's unnecessary delay, less
  // and the detection becomes inaccurate - at least with this autocorrelation
  // method. A spectral-domain method might need less than this, since
  // autocorrelation requires there to be at least two periods within the
  // window, against 1 for a spectrum reading.
  const auto windowSizeMs = 1000 * 5 / freq;
  return static_cast<int>(windowSizeMs * sampleRate / 1000);
}

std::vector<float> getAnalysisWindow(int windowSize) {
  std::vector<float> window((size_t)windowSize);
  const auto freq = twoPi / (float)windowSize;
  // TODO: make sure a rectangular window is tried.
  for (auto i = 0u; i < windowSize; ++i) {
    // A Hanning window.
    // For this use case and if there is not need for overlapping windows,
    // a flat-top might work as well.
    // window[i] = 1.f / fftSize;
    // i + 1 so that the tip of the window is at windowSize / 2, which is
    // convenient when taking the second half of it.
    window[i] = (1 - cosf((i + 1) * freq)) / 2;
  }
  return window;
}

void applyWindow(const std::vector<float> &window, std::vector<float> &input) {
  for (auto i = 0u; i < window.size(); ++i) {
    input[i] *= window[i];
  }
}

void getXCorr(pffft::Fft<float> &fft, std::vector<float> &time,
              const std::vector<float> &lpWindow,
              FormantShifterLoggerInterface &logger,
              pffft::Fft<float> *cepstrumFft = nullptr,
              const std::vector<float> *halfWindow = nullptr,
              Aligned<std::vector<float>> *cepstrum = nullptr) {
  Aligned<std::vector<std::complex<float>>> freq;
  freq.value.resize(fft.getSpectrumSize());
  auto freqData = freq.value.data();
  auto timeData = time.data();
  fft.forward(timeData, freqData);

  if (cepstrum) {
    Aligned<std::vector<std::complex<float>>> logMag;
    // The information we're interested is doesn't exceed 3kHz. Assuming 44.1k,
    // it means we can divide the fft size by about 16. But we will mirror it to
    // enhance the periodicity, so only by 8.
    const int copiedBins = halfWindow->size();
    const auto cepstrumSize = (copiedBins - 1) * 2;

    logMag.value.resize(cepstrumSize);

    auto halfWindowIt = halfWindow->begin();
    std::transform(freqData, freqData + copiedBins, logMag.value.begin(),
                   [&](const std::complex<float> &X) {
                     const auto power =
                         X.real() * X.real() + X.imag() * X.imag();
                     const auto w = *halfWindowIt++;
                     return std::complex<float>{w * FastLog2(power), 0.f};
                   });

    // Make periodic
    std::reverse_copy(logMag.value.begin() + 1,
                      logMag.value.begin() + copiedBins - 1,
                      logMag.value.begin() + copiedBins);

    logger.Log(logMag.value.data(), logMag.value.size(), "logMagSpectrum",
               [](const std::complex<float> &c) { return c.real(); });

    cepstrum->value.resize(cepstrumSize);
    cepstrumFft->inverse(logMag.value.data(), cepstrum->value.data());

    logger.Log(cepstrum->value.data(), cepstrum->value.size(), "cepstrum");
  }

  for (auto i = 0; i < lpWindow.size(); ++i) {
    auto &X = freqData[i];
    X *= lpWindow[i] * std::complex<float>{X.real(), -X.imag()};
  }
  std::fill(freqData + lpWindow.size(), freqData + fft.getSpectrumSize(), 0.f);
  fft.inverse(freqData, timeData);
  const auto normalizer = 1.f / timeData[0];
  for (auto i = 0u; i < fft.getLength(); ++i) {
    timeData[i] *= normalizer;
  }
}

std::vector<float> getLpWindow(int sampleRate, int fftSize) {
  std::vector<float> window(fftSize / 2);
  const int cutoffBin =
      std::min(fftSize / 2, fftSize * cutoffFreq / sampleRate);
  const int rollOffSize = fftSize * 200 / sampleRate;
  std::fill(window.begin(), window.begin() + cutoffBin, 1.f);
  for (auto i = 0; i < rollOffSize && cutoffBin + rollOffSize < fftSize / 2;
       ++i) {
    window[cutoffBin + i] = 1.f - i / static_cast<float>(rollOffSize);
  }
  std::fill(window.begin() + cutoffBin + rollOffSize, window.end(), 0.f);
  return window;
}

std::vector<float> getWindowXCorr(pffft::Fft<float> &fftEngine,
                                  const std::vector<float> &window,
                                  const std::vector<float> &lpWindow) {
  Aligned<std::vector<float>> xcorrAligned;
  auto &xcorr = xcorrAligned.value;
  xcorr.resize((fftEngine.getLength()));
  std::copy(window.begin(), window.end(), xcorr.begin());
  std::fill(xcorr.begin() + window.size(), xcorr.end(), 0.f);
  DummyFormantShifterLogger logger;
  getXCorr(fftEngine, xcorr, lpWindow, logger);
  return xcorr;
}

constexpr auto decimationFactor = 8;

constexpr auto getCepstrumSize(int fftSize) {
  return fftSize / decimationFactor;
}

constexpr auto getCopiedSize(int fftSize) {
  return fftSize / decimationFactor / 2 + 1;
}

std::vector<float> getHalfWindow(int fftSize) {
  std::vector<float> window = getAnalysisWindow(getCepstrumSize(fftSize));
  const auto copiedSize = getCopiedSize(fftSize);
  window.erase(window.begin(), window.begin() + window.size() - copiedSize);
  return window;
}
} // namespace

PitchDetectorImpl::PitchDetectorImpl(
    int sampleRate, const std::optional<float> &leastFrequencyToDetect,
    std::unique_ptr<FormantShifterLoggerInterface> logger)
    : _sampleRate(sampleRate), _logger(std::move(logger)),
      _window(getAnalysisWindow(
          getWindowSizeSamples(sampleRate, leastFrequencyToDetect))),
      _fftSize(getFftSizeSamples(static_cast<int>(_window.size()))),
      _fwdFft(_fftSize), _cepstrumFft(getCepstrumSize(_fftSize)),
      _halfWindow(getHalfWindow(_fftSize)),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _lastSearchIndex(
          std::min(_fftSize / 2, static_cast<int>(sampleRate / 70))),
      _windowXcor(getWindowXCorr(_fwdFft, _window, _lpWindow)) {

  // Fill the first ring buffer with half the window size of zeros.
  std::vector<float> zeros(_window.size() / 2);
  std::fill(zeros.begin(), zeros.end(), 0.f);
  _ringBuffers[0].writeBuff(zeros.data(), zeros.size());
}

std::optional<float> PitchDetectorImpl::process(const float *audio,
                                                int audioSize) {
  _ringBuffers[0].writeBuff(audio, audioSize);
  _ringBuffers[1].writeBuff(audio, audioSize);
  _logger->NewSamplesComing(audioSize);
  static auto count = 0;
  _logger->Log(44100, "sampleRate");
  _logger->Log(_fftSize, "fftSize");
  _logger->Log(_cepstrumFft.getLength(), "cepstrumFftSize");

  while (_ringBuffers[_ringBufferIndex].readAvailable() >= _window.size()) {
    std::vector<float> time(_fftSize);
    _ringBuffers[_ringBufferIndex].readBuff(time.data(), _window.size());
    std::fill(time.begin() + _window.size(), time.begin() + _fftSize, 0.f);
    _logger->Log(time.data(), time.size(), "inputAudio");
    applyWindow(_window, time);
    Aligned<std::vector<float>> cepstrum;
    _logger->Log(time.data(), time.size(), "windowedAudio");
    getXCorr(_fwdFft, time, _lpWindow, *_logger, &_cepstrumFft, &_halfWindow,
             &cepstrum);
    _logger->ProcessFinished(nullptr, 0);

    // We're using this for a tuner, so look between 60Hz and 500Hz.
    constexpr auto maxPeriod = 1 / 60.f;
    constexpr auto minPeriod = 1 / 500.f;
    const auto cepstrumSamplePeriod =
        decimationFactor / static_cast<float>(_sampleRate);
    const auto firstCepstrumSample =
        static_cast<int>(minPeriod / cepstrumSamplePeriod);
    const auto lastCepstrumSample = std::min<int>(
        maxPeriod / cepstrumSamplePeriod, cepstrum.value.size() / 2);
    const auto it =
        std::max_element(cepstrum.value.begin() + firstCepstrumSample,
                         cepstrum.value.begin() + lastCepstrumSample);
    const auto maxCepstrumIndex = std::distance(cepstrum.value.begin(), it);
    const auto cepstrumEstimateHz =
        1 / (maxCepstrumIndex * cepstrumSamplePeriod);

    auto &max = _maxima[_ringBufferIndex] = 0;
    auto maxIndex = 0;
    auto wentNegative = false;
    for (auto i = 0; i < _lastSearchIndex; ++i) {
      wentNegative |= time[i] < 0;
      if (wentNegative && time[i] > max) {
        max = time[i];
        maxIndex = i;
      }
    }
    max /= _windowXcor[maxIndex];
    if (max > 0.9) {
      // _detectedPitch = _sampleRate / maxIndex;
      _detectedPitch = cepstrumEstimateHz;
    } else {
      _detectedPitch.reset();
    }
    _ringBufferIndex = (_ringBufferIndex + 1) % _ringBuffers.size();
  }
  return _detectedPitch;
}
} // namespace saint
