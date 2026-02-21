#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace saint {

enum class FilterType { Lowpass, Highpass };

template <int P>
struct Coefs {
    std::array<double, P + 1> b;  // Numerator coefficients (b[0]...b[P])
    std::array<double, P + 1> a;  // Denominator coefficients (a[0]...a[P]), a[0] = 1
};

template <int P>
Coefs<P> butterworthCoefs(FilterType type, double cutoffHz, int sampleRate) {
    // AI-generated, checked against scipy output and testing.

    Coefs<P> coefs = {};
    coefs.b.fill(0.0);
    coefs.a.fill(0.0);

    const double fs = static_cast<double>(sampleRate);

    // Calculate digital frequency (radians)
    const double omega_digital = 2.0 * M_PI * cutoffHz / fs;

    // Pre-warp for bilinear transform with κ = 2
    // Analog frequency: Ω = 2 * tan(ω/2)
    const double omega_analog = 2.0 * std::tan(omega_digital / 2.0);

    // Compute analog low-pass prototype poles on unit circle
    // Butterworth poles: exp(j * pi * (2k + P + 1) / (2P)) for k = 0, 1, ..., P-1
    std::vector<std::complex<double>> prototypePoles;
    for (int k = 0; k < P; ++k) {
        double angle = M_PI * (2.0 * k + P + 1.0) / (2.0 * P);
        prototypePoles.emplace_back(std::cos(angle), std::sin(angle));
    }

    // Transform analog prototype poles to desired filter type
    std::vector<std::complex<double>> analogPoles;
    for (const auto& pole : prototypePoles) {
        if (type == FilterType::Lowpass) {
            // Scale by analog cutoff frequency
            analogPoles.push_back(pole * omega_analog);
        } else {  // Highpass
            // Highpass transform: s -> Ω_c / s
            analogPoles.push_back(omega_analog / pole);
        }
    }

    // Apply bilinear transform to convert to digital domain
    // bilinear: s -> 2*(z-1)/(z+1), solving for z: z = (2 + s)/(2 - s)
    std::vector<std::complex<double>> digitalPoles;
    for (const auto& sp : analogPoles) {
        digitalPoles.push_back((2.0 + sp) / (2.0 - sp));
    }

    // Build the denominator polynomial from poles
    // Denominator = product of (1 - pole_k * z^-1)
    // Which in z-domain is: product of (z - pole_k)
    std::vector<std::complex<double>> denPoly = {1.0};
    for (const auto& pole : digitalPoles) {
        std::vector<std::complex<double>> newPoly(denPoly.size() + 1, 0.0);
        for (size_t i = 0; i < denPoly.size(); ++i) {
            newPoly[i] += denPoly[i];
            newPoly[i + 1] -= denPoly[i] * pole;
        }
        denPoly = std::move(newPoly);
    }

    // Copy denominator coefficients (imaginary parts should be ~0)
    for (int i = 0; i <= P; ++i) {
        coefs.a[i] = denPoly[i].real();
    }

    // Build numerator based on filter type
    if (type == FilterType::Lowpass) {
        // Lowpass: H(z) = K * (1 + z^-1)^P
        // In z-domain: K * (z + 1)^P / z^P
        std::vector<std::complex<double>> numPoly = {1.0};
        for (int k = 0; k < P; ++k) {
            std::vector<std::complex<double>> newPoly(numPoly.size() + 1, 0.0);
            for (size_t i = 0; i < numPoly.size(); ++i) {
                newPoly[i] += numPoly[i];
                newPoly[i + 1] += numPoly[i];
            }
            numPoly = std::move(newPoly);
        }
        for (int i = 0; i <= P; ++i) {
            coefs.b[i] = numPoly[i].real();
        }
    } else {  // Highpass
        // Highpass: H(z) = K * (1 - z^-1)^P
        // In z-domain: K * (z - 1)^P / z^P
        std::vector<std::complex<double>> numPoly = {1.0};
        for (int k = 0; k < P; ++k) {
            std::vector<std::complex<double>> newPoly(numPoly.size() + 1, 0.0);
            for (size_t i = 0; i < numPoly.size(); ++i) {
                newPoly[i] += numPoly[i];
                newPoly[i + 1] -= numPoly[i];
            }
            numPoly = std::move(newPoly);
        }
        for (int i = 0; i <= P; ++i) {
            coefs.b[i] = numPoly[i].real();
        }
    }

    // Normalize to get unity gain at DC (lowpass) or Nyquist (highpass)
    double numSum = 0.0;
    double denSum = 0.0;
    if (type == FilterType::Lowpass) {
        // Evaluate H(z) at z = 1 (DC)
        for (int i = 0; i <= P; ++i) {
            numSum += coefs.b[i];
            denSum += coefs.a[i];
        }
    } else {  // Highpass
        // Evaluate H(z) at z = -1 (Nyquist)
        for (int i = 0; i <= P; ++i) {
            numSum += coefs.b[i] * (i % 2 == 0 ? 1.0 : -1.0);
            denSum += coefs.a[i] * (i % 2 == 0 ? 1.0 : -1.0);
        }
    }
    double gain = numSum / denSum;
    for (int i = 0; i <= P; ++i) {
        coefs.b[i] /= gain;
    }

    return coefs;
}

template <int FilterOrder>
class ButterworthFilter {
   public:
    ButterworthFilter(int numChannels, int channel, Coefs<FilterOrder> coeffs)
        : _numChannels{numChannels},
          _channel{channel},
          _b(std::move(coeffs.b)),
          _a(std::move(coeffs.a)),
          _x{},  // Zero-initialize delay lines
          _y{} {}

    void process(float* data, int samplesPerChannel) {
        for (int i = 0; i < samplesPerChannel; ++i) {
            const auto j = i * _numChannels + _channel;
            const auto input = data[j];
            const auto output = process(input);
            data[j] = output;
        }
    }

    float process(float input) {
        // Direct Form I implementation
        // y[n] = (b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] ...) / a[0]

        // Shift delay lines to make room for new sample
        // After shift: _x[0] will hold x[n], _x[1] holds x[n-1], etc.
        for (size_t i = _x.size() - 1; i > 0; --i) {
            _x[i] = _x[i - 1];
            _y[i] = _y[i - 1];
        }

        // Store new input
        _x[0] = input;

        // Compute output
        double output = 0.0;
        for (size_t i = 0; i < _b.size(); ++i) {
            output += _b[i] * _x[i];
        }
        for (size_t i = 1; i < _a.size(); ++i) {
            output -= _a[i] * _y[i];
        }
        output /= _a[0];

        // Store new output
        _y[0] = output;

        return static_cast<float>(output);
    }

   private:
    const int _numChannels;
    const int _channel;
    std::array<double, FilterOrder + 1> _b;
    std::array<double, FilterOrder + 1> _a;
    std::array<double, FilterOrder + 1> _x;  // input delay line
    std::array<double, FilterOrder + 1> _y;  // output delay line
};
}  // namespace saint