#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "PitchDetector.h"
#include "PitchDetectorTypes.h"

#include <onnxruntime_cxx_api.h>

namespace saint {

// Benchmark-only wrapper around a PESTO streaming ONNX model
// (https://github.com/SonyCSLParis/pesto, exported with realtime.export_onnx).
// Not part of the production library.
//
// The model must have been exported with chunk size == blockSize and the
// matching sample rate (both are frozen at export time), so each process()
// call feeds exactly one chunk and no FIFO re-chunking is needed.
class PestoPitchDetector : public PitchDetector {
   public:
    PestoPitchDetector(const std::filesystem::path& modelPath, int sampleRate,
                       ChannelFormat channelFormat, int blockSize, float confidenceThreshold);

    float process(const float* input, DebugOutput* debugOutput = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const int _blockSize;
    const int _numChannels;
    const float _confidenceThreshold;
    Ort::Session _session;
    Ort::MemoryInfo _memoryInfo;
    std::vector<float> _monoBuffer;
    std::vector<float> _cache;  // streaming state, re-fed on every call
};

}  // namespace saint
