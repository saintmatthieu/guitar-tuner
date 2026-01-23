/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  DummyPitchDetectorLogger.h

  A class for shifting the formants of a voice signal.

  Matthieu Hodgkinson

**********************************************************************/
#pragma once

#include "PitchDetectorLoggerInterface.h"

namespace saint {
class DummyPitchDetectorLogger : public PitchDetectorLoggerInterface {
public:
  ~DummyPitchDetectorLogger() override;
  void SamplesRead(int) override {}
  bool StartNewEstimate() override { return false; }
  void Log(int value, const char *name) const override;
  void Log(const float *samples, size_t size, const char *name) const override;
  void Log(const std::complex<float> *samples, size_t size, const char *name,
           const std::function<float(const std::complex<float> &)> &transform)
      const override;
  void EndNewEstimate(std::complex<float> *spectrum, size_t fftSize) override;
};
} // namespace saint
