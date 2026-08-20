#pragma once

#include <complex>
#include <vector>

#include "RealFft.h"

namespace saint {
// Autocorrelation of the frame whose spectrum is `freq`, band-limited by `lpWindow` and
// normalised so that lag 0 reads 1. `freq` is consumed.
void getXCorr(RealFft&, std::vector<float>& time, std::vector<std::complex<float>>& freq,
              const std::vector<float>& lpWindow);

// The analysis window's own autocorrelation, which is what the overlap at a given lag has to be
// measured against.
std::vector<float> getWindowXCorr(RealFft&, const std::vector<float>& window,
                                  const std::vector<float>& lpWindow);

struct AutocorrPeak {
    int lag = 0;
    // The autocorrelation at `lag`, corrected for the window overlap left there: 1 for a
    // perfectly periodic frame, 0 for noise. This is the presence score.
    float presence = 0.f;
};

// The most periodic lag in [firstLag, lastLag), and how periodic. Lags before the
// autocorrelation first goes negative are skipped, so the lag-0 lobe cannot win.
AutocorrPeak findAutocorrPeak(const std::vector<float>& xcorr,
                              const std::vector<float>& windowXcorr, int firstLag, int lastLag);
}  // namespace saint
