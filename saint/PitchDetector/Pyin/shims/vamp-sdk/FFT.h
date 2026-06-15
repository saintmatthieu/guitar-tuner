// Minimal, self-contained replacement for Vamp::FFTReal, the only symbol the
// vendored pYIN core (Yin/YinUtil) pulls from the Vamp plugin SDK.
//
// pYIN's YinUtil declares `Vamp::FFTReal m_fft;` and uses exactly two calls:
//   forward(const double* ri, double* co)  -- real -> interleaved complex (n/2+1 bins)
//   inverse(const double* ci, double* ro)  -- interleaved complex -> real, scaled 1/n
// (see the Vamp SDK's vamp-sdk/FFT.h for the contract). Providing this header on
// the include path lets us compile the pYIN algorithm core without vendoring the
// Vamp SDK. This is our own (double-precision) implementation, not Vamp code.
//
// This file is part of the SAINT guitar-tuner project (MIT). It only mimics the
// Vamp::FFTReal *interface*; it shares no code with the Vamp SDK.
#pragma once

#include <vector>

namespace Vamp {

// Real-to-complex / complex-to-real FFT for sizes that are a power of two.
// The forward transform is unscaled; the inverse is scaled by 1/n. Output of
// forward / input of inverse is the half-spectrum: n/2+1 interleaved (re, im)
// pairs, i.e. n+2 doubles.
class FFTReal {
   public:
    explicit FFTReal(unsigned int n);

    // ri: n real samples. co: (n/2+1) interleaved complex bins (n+2 doubles).
    void forward(const double* ri, double* co);

    // ci: (n/2+1) interleaved complex bins (n+2 doubles). ro: n real samples,
    // scaled by 1/n (only the real part of the inverse transform is returned).
    void inverse(const double* ci, double* ro);

   private:
    unsigned int m_n;
    std::vector<double> m_scratch;  // 2*n interleaved-complex work buffer
};

}  // namespace Vamp
