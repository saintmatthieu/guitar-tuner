#include "vamp-sdk/FFT.h"

#include <cmath>
#include <utility>

namespace Vamp {
namespace {

constexpr double kPi = 3.14159265358979323846;

// In-place iterative radix-2 Cooley-Tukey FFT over an interleaved [re, im] array
// of length n (n a power of two). sign = -1 for the forward transform, +1 for the
// inverse. No scaling is applied here.
void fftRadix2(double* a, unsigned int n, int sign) {
    // Bit-reversal permutation.
    for (unsigned int i = 1, j = 0; i < n; ++i) {
        unsigned int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[2 * i], a[2 * j]);
            std::swap(a[2 * i + 1], a[2 * j + 1]);
        }
    }

    for (unsigned int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * kPi / static_cast<double>(len);
        const double wlenRe = std::cos(ang);
        const double wlenIm = std::sin(ang);
        for (unsigned int i = 0; i < n; i += len) {
            double wRe = 1.0;
            double wIm = 0.0;
            for (unsigned int k = 0; k < len / 2; ++k) {
                const unsigned int e = 2 * (i + k);
                const unsigned int o = 2 * (i + k + len / 2);
                const double uRe = a[e];
                const double uIm = a[e + 1];
                const double vRe = a[o] * wRe - a[o + 1] * wIm;
                const double vIm = a[o] * wIm + a[o + 1] * wRe;
                a[e] = uRe + vRe;
                a[e + 1] = uIm + vIm;
                a[o] = uRe - vRe;
                a[o + 1] = uIm - vIm;
                const double nwRe = wRe * wlenRe - wIm * wlenIm;
                wIm = wRe * wlenIm + wIm * wlenRe;
                wRe = nwRe;
            }
        }
    }
}

}  // namespace

FFTReal::FFTReal(unsigned int n) : m_n(n), m_scratch(2u * n) {}

void FFTReal::forward(const double* ri, double* co) {
    const unsigned int n = m_n;
    for (unsigned int i = 0; i < n; ++i) {
        m_scratch[2 * i] = ri[i];
        m_scratch[2 * i + 1] = 0.0;
    }
    fftRadix2(m_scratch.data(), n, -1);
    for (unsigned int k = 0; k <= n / 2; ++k) {
        co[2 * k] = m_scratch[2 * k];
        co[2 * k + 1] = m_scratch[2 * k + 1];
    }
}

void FFTReal::inverse(const double* ci, double* ro) {
    const unsigned int n = m_n;
    // Lower half-spectrum (bins 0..n/2) comes straight from the input.
    for (unsigned int k = 0; k <= n / 2; ++k) {
        m_scratch[2 * k] = ci[2 * k];
        m_scratch[2 * k + 1] = ci[2 * k + 1];
    }
    // Upper half is the conjugate mirror of the lower half (Hermitian symmetry).
    for (unsigned int k = n / 2 + 1; k < n; ++k) {
        m_scratch[2 * k] = ci[2 * (n - k)];
        m_scratch[2 * k + 1] = -ci[2 * (n - k) + 1];
    }
    fftRadix2(m_scratch.data(), n, +1);
    const double scale = 1.0 / static_cast<double>(n);
    for (unsigned int i = 0; i < n; ++i) {
        ro[i] = m_scratch[2 * i] * scale;
    }
}

}  // namespace Vamp
