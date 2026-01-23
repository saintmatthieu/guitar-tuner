#include "PitchDetectorImpl.h"
#include "DummyFormantShifterLogger.h"
#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <math.h>
#include <optional>

namespace saint {

std::unique_ptr<PitchDetector> PitchDetector::createInstance(int sampleRate) {
  return std::make_unique<PitchDetectorImpl>(
      sampleRate, std::make_unique<DummyFormantShifterLogger>());
}

namespace {
constexpr auto cutoffFreq = 1500;

constexpr float log2ToDb = 20 / 3.321928094887362f;
constexpr float leastFrequencyToDetect = /* A0 */ 27.5f;

int getFftOrder(int windowSize) {
  return static_cast<int>(ceilf(log2f((float)windowSize)));
}

int getFftSizeSamples(int windowSize) { return 1 << getFftOrder(windowSize); }

int getWindowSizeSamples(int sampleRate) {
  // 3.3 times the fundamental period. More and that's unnecessary delay, less
  // and the detection becomes inaccurate - at least with this autocorrelation
  // method. A spectral-domain method might need less than this, since
  // autocorrelation requires there to be at least two periods within the
  // window, against 1 for a spectrum reading.
  const auto windowSizeMs = 1000 * 3.3 / leastFrequencyToDetect;
  return static_cast<int>(windowSizeMs * sampleRate / 1000);
}

void applyWindow(const std::vector<float> &window, std::vector<float> &input) {
  for (auto i = 0u; i < window.size(); ++i) {
    input[i] *= window[i];
  }
}

void getXCorr(RealFft &fft, std::vector<float> &time,
              const std::vector<float> &lpWindow,
              FormantShifterLoggerInterface &logger,
              CepstrumData *cepstrumData = nullptr) {
  Aligned<std::vector<std::complex<float>>> freq;
  freq.value.resize(fft.size / 2);
  auto freqData = freq.value.data();
  auto timeData = time.data();

  fft.forward(timeData, freqData);

  if (cepstrumData) {
    takeCepstrum(freq.value.data(), freq.value.size(), *cepstrumData, logger);
  }

  for (auto i = 0; i < lpWindow.size(); ++i) {
    auto &X = freqData[i];
    X *= lpWindow[i] * std::complex<float>{X.real(), -X.imag()};
  }
  std::fill(freqData + lpWindow.size(), freqData + fft.size / 2, 0.f);
  fft.inverse(freqData, timeData);
  const auto normalizer = 1.f / timeData[0];
  for (auto i = 0; i < fft.size; ++i) {
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

std::vector<float> getWindowXCorr(RealFft &fft,
                                  const std::vector<float> &window,
                                  const std::vector<float> &lpWindow) {
  Aligned<std::vector<float>> xcorrAligned;
  auto &xcorr = xcorrAligned.value;
  xcorr.resize(fft.size);
  std::copy(window.begin(), window.end(), xcorr.begin());
  std::fill(xcorr.begin() + window.size(), xcorr.end(), 0.f);
  DummyFormantShifterLogger logger;
  getXCorr(fft, xcorr, lpWindow, logger);
  return xcorr;
}

double getCepstrumPeakFrequency(const CepstrumData &cepstrumData,
                                int sampleRate) {
  // We're using this for a tuner, so look between 30Hz (a detuned E0 on a bass
  // guitar) and 500Hz (Ukulele high A is 440Hz)
  constexpr auto maxPeriod = 1 / 30.f;
  constexpr auto minPeriod = 1 / 500.f;
  const auto firstCepstrumSample = static_cast<int>(minPeriod * sampleRate);
  const auto &vec = cepstrumData.vec();
  const auto lastCepstrumSample =
      std::min<int>(maxPeriod * sampleRate, vec.size() / 2);
  const auto it = std::max_element(vec.begin() + firstCepstrumSample,
                                   vec.begin() + lastCepstrumSample);
  const auto maxCepstrumIndex = std::distance(vec.begin(), it);
  if (maxCepstrumIndex == 0)
    return 0;
  else if (maxCepstrumIndex == vec.size())
    return sampleRate / maxCepstrumIndex;

  // Parabolic interpolation
  const auto prev = vec[maxCepstrumIndex - 1];
  const auto max = vec[maxCepstrumIndex];
  const auto next = vec[maxCepstrumIndex + 1];
  const auto p = 0.5f * (prev - next) / (prev + next - 2 * max);
  const auto peakIndex = maxCepstrumIndex + p;
  return sampleRate / peakIndex;
}
} // namespace

PitchDetectorImpl::PitchDetectorImpl(
    int sampleRate, std::unique_ptr<FormantShifterLoggerInterface> logger)
    : _sampleRate(sampleRate), _logger(std::move(logger)),
      _window(utils::getAnalysisWindow(getWindowSizeSamples(sampleRate))),
      _fftSize(getFftSizeSamples(static_cast<int>(_window.size()))),
      _fwdFft(_fftSize), _cepstrumData(_fftSize),
      _lpWindow(getLpWindow(sampleRate, _fftSize)),
      _lastSearchIndex(std::min(
          _fftSize / 2, static_cast<int>(sampleRate / leastFrequencyToDetect))),
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
  _logger->Log(_sampleRate, "sampleRate");
  _logger->Log(_fftSize, "fftSize");
  _logger->Log(_cepstrumData.fft.size, "cepstrumFftSize");

  std::optional<float> result;

  while (_ringBuffers[_ringBufferIndex].readAvailable() >= _window.size()) {
    std::vector<float> time(_fftSize);
    _ringBuffers[_ringBufferIndex].readBuff(time.data(), _window.size());
    std::fill(time.begin() + _window.size(), time.begin() + _fftSize, 0.f);
    _logger->Log(time.data(), time.size(), "inputAudio");
    applyWindow(_window, time);
    _logger->Log(time.data(), time.size(), "windowedAudio");
    getXCorr(_fwdFft, time, _lpWindow, *_logger, &_cepstrumData);
    _logger->Log(time.data(), time.size(), "xcorr");
    _logger->ProcessFinished(nullptr, 0);

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
    _ringBufferIndex = (_ringBufferIndex + 1) % _ringBuffers.size();

    max /= _windowXcor[maxIndex];
    if (max > 0.9) {
      // _detectedPitch = _sampleRate / maxIndex;
      result = getCepstrumPeakFrequency(_cepstrumData, _sampleRate);
    } else if (!result.has_value()) {
      result = 0.f;
    }
  }

  return result;
}
} // namespace saint
