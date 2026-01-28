/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  PitchDetectorLogger.cpp

  Matthieu Hodgkinson

**********************************************************************/
#include "PitchDetectorLogger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "testUtils.h"

namespace saint {
namespace {
template <typename Iterator>
void PrintPythonVector(std::ofstream& ofs, Iterator begin, Iterator end, const char* name) {
    ofs << name << " = [";
    std::for_each(begin, end, [&](float x) { ofs << x << ","; });
    ofs << "]\n";
}
}  // namespace

PitchDetectorLogger::PitchDetectorLogger(int sampleRate, int estimateIndex)
    : mSampleRate{sampleRate}, mEstimateIndex{estimateIndex} {}

PitchDetectorLogger::~PitchDetectorLogger() {}

void PitchDetectorLogger::SamplesRead(int count) {
    mRingBufferCount += count;
}

bool PitchDetectorLogger::StartNewEstimate() {
    const auto ok = mEstimateCount++ == mEstimateIndex;
    if (ok) {
        // Ready for logging.
        mAnalysisSampleIndex = mRingBufferCount;
        const auto file = testUtils::getOutDir() / "PitchDetectorLog.py";
        std::cout << "Logging PitchDetector analysis to " << file << "\n";
        mOfs = std::make_unique<std::ofstream>(file);
        *mOfs << "sampleRate = " << mSampleRate << "\n";
        *mOfs << "audioIndex = " << *mAnalysisSampleIndex << "\n";
    }
    return ok;
}

void PitchDetectorLogger::Log(int value, const char* name) const {
    if (mOfs) {
        *mOfs << name << " = " << value << "\n";
    }
}

void PitchDetectorLogger::Log(const float* samples, size_t size, const char* name) const {
    if (!mOfs) {
        // Keep it lightweight if we're not logging.
        return;
    }
    assert(std::all_of(samples, samples + size, [](float x) { return std::isfinite(x); }));
    PrintPythonVector(*mOfs, samples, samples + size, name);
}

void PitchDetectorLogger::Log(
    const std::complex<float>* cv, size_t cvSize, const char* name,
    const std::function<float(const std::complex<float>&)>& transform) const {
    if (!mOfs) {
        return;
    }
    assert(std::all_of(cv, cv + cvSize, [](const std::complex<float>& x) {
        return std::isfinite(x.real()) && std::isfinite(x.imag());
    }));
    std::vector<float> v(cvSize);
    std::transform(cv, cv + cvSize, v.begin(), transform);
    PrintPythonVector(*mOfs, v.begin(), v.end(), name);
}

void PitchDetectorLogger::EndNewEstimate(std::complex<float>* spectrum, size_t fftSize) {
    if (!mOfs) {
        return;
    }
    // Such a spectrum of only (1 + 0j) is that of a click, which should be
    // audible ...
    // std::fill(spectrum, spectrum + fftSize / 2 + 1, 1.f);
    mOfs.reset();
}
}  // namespace saint
