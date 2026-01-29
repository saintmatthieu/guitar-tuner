/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  DummyPitchDetectorLogger.cpp

  A class for shifting the formants of a voice signal.

  Matthieu Hodgkinson

**********************************************************************/
#include "DummyPitchDetectorLogger.h"

namespace saint {
DummyPitchDetectorLogger::~DummyPitchDetectorLogger() {}

void DummyPitchDetectorLogger::Log(int value, const char* name) const {}

void DummyPitchDetectorLogger::Log(const float* samples, size_t size, const char* name,
                                   const std::function<float(float)>& transform) const {}

void DummyPitchDetectorLogger::Log(
    const std::complex<float>* samples, size_t size, const char* name,
    const std::function<float(const std::complex<float>&)>& transform) const {}

void DummyPitchDetectorLogger::EndNewEstimate(std::complex<float>* spectrum, size_t fftSize) {}
}  // namespace saint
