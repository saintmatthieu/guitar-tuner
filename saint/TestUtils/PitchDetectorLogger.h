/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  PitchDetectorLogger.h

  Implements PitchDetectorLoggerInterface, and also provides tuning utilities
  to override algorithm parameters.

  Matthieu Hodgkinson

**********************************************************************/
#pragma once

#include "PitchDetector/PitchDetectorLoggerInterface.h"
#include <fstream>
#include <map>
#include <memory>
#include <optional>

namespace saint {
class PitchDetectorLogger : public PitchDetectorLoggerInterface {
public:
  PitchDetectorLogger(int sampleRate, int estimateIndex);
  ~PitchDetectorLogger() override;

  void SamplesRead(int count) override;
  bool StartNewEstimate() override;
  void Log(int value, const char *name) const override;
  void Log(const float *samples, size_t size, const char *name) const override;
  void Log(const std::complex<float> *samples, size_t size, const char *name,
           const std::function<float(const std::complex<float> &)> &transform)
      const override;
  /*!
   * @brief If not already, disables the logging and marks the spectrum with an
   * audible event to make clear where in the signal the logging took place.
   * (Of course not for use in production :D)
   */
  void EndNewEstimate(std::complex<float> *spectrum, size_t fftSize) override;

  std::optional<int> analysisAudioIndex() const { return mAnalysisSampleIndex; }

private:
  const int mSampleRate;
  const int mEstimateIndex;
  std::unique_ptr<std::ofstream> mOfs;
  int mEstimateCount = 0;
  int mRingBufferCount = 0;
  std::optional<int> mAnalysisSampleIndex;
};
} // namespace saint
