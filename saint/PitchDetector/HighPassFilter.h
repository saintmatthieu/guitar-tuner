#include <algorithm>
#include <vector>

namespace saint {
class Filter {
   public:
    Filter(int numChannels, int channel, std::vector<double> b, std::vector<double> a)
        : _numChannels{numChannels},
          _channel{channel},
          _b(std::move(b)),
          _a(std::move(a)),
          _x(_b.size(), 0.f),
          _y(_a.size(), 0.f) {}

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
        // y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - a2*y[n-2] ...
        _x[0] = input;
        double output = 0.;
        for (size_t i = 0; i < _b.size(); ++i) {
            output += _b[i] * _x[i];
        }
        for (size_t i = 1; i < _a.size(); ++i) {
            output -= _a[i] * _y[i];
        }
        output /= _a[0];
        _y[0] = output;
        // Shift delay lines
        for (size_t i = _x.size() - 1; i > 0; --i) {
            _x[i] = _x[i - 1];
            _y[i] = _y[i - 1];
        }
        return output;
    }

   private:
    const int _numChannels;
    const int _channel;
    std::vector<double> _b;
    std::vector<double> _a;
    std::vector<double> _x;  // input delay line
    std::vector<double> _y;  // output delay line
};

class HighPassFilter : public Filter {
   public:
    HighPassFilter(int numChannels, int channel)
        : Filter(numChannels, channel,
                 // Cutoff 1kHz
                 {0.8671035126423323, -2.601310537926997, 2.601310537926997, -0.8671035126423323},
                 {1.0, -2.7152853556329544, 2.4696743431401167, -0.7518684023655857}) {}
};

class LowPassFilter : public Filter {
   public:
    LowPassFilter(int numChannels, int channel)
        : Filter(numChannels, channel,
                 // Cutoff 5kHz
                 {0.08315986992995228, 0.16631973985990456, 0.08315986992995228},
                 {1.0, -1.035171209738942, 0.3678106894587511}) {}
};
}  // namespace saint