#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace saint {

enum class WindowType { Hamming, Blackman, Kaiser };

/**
 * Design FIR lowpass filter coefficients using windowed-sinc method.
 * The resulting filter has linear phase response.
 *
 * @param numTaps Number of filter taps (should be odd for symmetric Type I FIR)
 * @param cutoffHz Cutoff frequency in Hz
 * @param sampleRate Sample rate in Hz
 * @param window Window function to apply
 * @return Vector of FIR coefficients (symmetric, linear phase)
 */
template <int NumTaps>
std::array<double, NumTaps> designFIRLowpass(double cutoffHz, int sampleRate,
                                             WindowType window = WindowType::Blackman) {
    static_assert(NumTaps % 2 == 1, "NumTaps should be odd for symmetric linear-phase FIR");

    std::array<double, NumTaps> h;
    const int M = NumTaps - 1;                // Filter order
    const double fc = cutoffHz / sampleRate;  // Normalized cutoff frequency

    // Generate ideal sinc-based lowpass filter (time domain)
    for (int n = 0; n < NumTaps; ++n) {
        const int m = n - M / 2;  // Center the filter
        if (m == 0) {
            h[n] = 2.0 * fc;  // sinc(0) = 1
        } else {
            h[n] = std::sin(2.0 * M_PI * fc * m) / (M_PI * m);
        }
    }

    // Apply window function
    for (int n = 0; n < NumTaps; ++n) {
        double w = 1.0;
        switch (window) {
            case WindowType::Hamming: {
                // Hamming window
                w = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / M);
                break;
            }
            case WindowType::Blackman: {
                // Blackman window (better stopband attenuation)
                const double a0 = 0.42;
                const double a1 = 0.5;
                const double a2 = 0.08;
                w = a0 - a1 * std::cos(2.0 * M_PI * n / M) + a2 * std::cos(4.0 * M_PI * n / M);
                break;
            }
            case WindowType::Kaiser: {
                // Simplified Kaiser window (beta = 5.0)
                // For more precise implementation, would need modified Bessel function
                // Using Hamming as approximation for now
                w = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / M);
                break;
            }
        }
        h[n] *= w;
    }

    // Normalize to unity gain at DC
    double sum = 0.0;
    for (int n = 0; n < NumTaps; ++n) {
        sum += h[n];
    }
    for (int n = 0; n < NumTaps; ++n) {
        h[n] /= sum;
    }

    return h;
}

/**
 * Linear-phase FIR filter implementation.
 * Uses symmetric coefficients for guaranteed linear phase response.
 */
template <int NumTaps>
class FIRFilter {
    static_assert(NumTaps % 2 == 1, "NumTaps should be odd for symmetric linear-phase FIR");

   public:
    int delaySamples() const {
        return (NumTaps - 1) / 2;  // Group delay for linear phase FIR
    }

    FIRFilter(int numChannels, int channel, std::array<double, NumTaps> coefficients)
        : _numChannels{numChannels},
          _channel{channel},
          _coeffs(std::move(coefficients)),
          _delayLine{},  // Zero-initialize
          _writeIndex{0} {}

    void process(float* data, int samplesPerChannel) {
        for (int i = 0; i < samplesPerChannel; ++i) {
            const auto j = i * _numChannels + _channel;
            const auto input = data[j];
            const auto output = process(input);
            data[j] = output;
        }
    }

    float process(float input) {
        // Add new sample to circular delay line
        _delayLine[_writeIndex] = input;

        // Compute FIR output using Direct Form
        double output = 0.0;
        int readIndex = _writeIndex;

        for (int i = 0; i < NumTaps; ++i) {
            output += _coeffs[i] * _delayLine[readIndex];

            // Move backwards through circular buffer
            if (readIndex == 0) {
                readIndex = NumTaps - 1;
            } else {
                --readIndex;
            }
        }

        // Advance write index
        ++_writeIndex;
        if (_writeIndex >= NumTaps) {
            _writeIndex = 0;
        }

        return static_cast<float>(output);
    }

    /**
     * Reset filter state (clear delay line)
     */
    void reset() {
        _delayLine.fill(0.0);
        _writeIndex = 0;
    }

   private:
    const int _numChannels;
    const int _channel;
    std::array<double, NumTaps> _coeffs;
    std::array<double, NumTaps> _delayLine;
    int _writeIndex;
};

}  // namespace saint
