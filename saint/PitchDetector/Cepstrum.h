#pragma once

#include <charconv>

#include "RealFft.h"

namespace saint {
class PitchDetectorLoggerInterface;

class CepstrumData {
   public:
    CepstrumData(int fftSize);

    RealFft fft;
    const std::vector<float> halfWindow;

    std::vector<float>& vec() {
        return _cepstrum.value;
    }
    const std::vector<float>& vec() const {
        return _cepstrum.value;
    }
    float* ptr() {
        return _cepstrum.value.data();
    }
    Aligned<std::vector<float>>& aligned() {
        return _cepstrum;
    }

   private:
    Aligned<std::vector<float>> _cepstrum;
};

/**
 * @param spectrum N/2 complex values, the first of which is DC + Nyquist
 */
void toCepstrum(const std::vector<std::complex<float>>& spectrum, CepstrumData& cepstrumData,
                PitchDetectorLoggerInterface& logger);

void toCepstrum(const std::vector<float>& logSpectrum, RealFft& fft,
                Aligned<std::vector<float>>& cepstrumAligned);

/**
 * @param cepstrumData
 * @return std::vector<float> Real-valued spectrum reconstructed from the cepstrum
 */
std::vector<float> fromCepstrum(RealFft& fft, const float* cepstrumPtr);
}  // namespace saint