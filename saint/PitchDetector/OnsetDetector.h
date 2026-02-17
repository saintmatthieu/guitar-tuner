#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "HighPassFilter.h"
#include "PitchDetectorTypes.h"

namespace saint {

class OnsetDetector {
   public:
    OnsetDetector(int sampleRate, ChannelFormat channelFormat, int samplesPerBlockPerChannel,
                  float minFreq);

    using DebugOutput = std::unordered_map<std::string, float>;

    bool process(const float* audio, DebugOutput* = nullptr);
    bool process(float* audio, DebugOutput* = nullptr);

   private:
    double updatePowerAverage(double newPower);

    const ChannelFormat _channelFormat;
    const int _blockSize;
    const std::vector<double> _window;
    std::vector<float> _audioBuffer;
    std::optional<double> _prevPower;
    const int _avgFilterLength;
    std::vector<double> _pastPowers;
    const std::vector<double> _avgWindow;
    const double _alpha;
};

}  // namespace saint
