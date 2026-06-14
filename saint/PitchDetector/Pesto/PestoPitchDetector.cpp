#include "PestoPitchDetector.h"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

namespace saint {
namespace {

Ort::Env& getOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PestoBenchmark");
    return env;
}

Ort::Session makeSession(const std::filesystem::path& modelPath) {
    Ort::SessionOptions options;
    // The benchmark already runs one detector per hardware thread; don't let
    // each session spawn its own thread pool on top of that.
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    return Ort::Session(getOrtEnv(), modelPath.c_str(), options);
}

// Model inputs/outputs as produced by pesto's realtime/export_onnx.py.
constexpr auto kAudioInput = "audio";
constexpr auto kCacheInput = "cache";
constexpr auto kPredictionOutput = "prediction";
constexpr auto kConfidenceOutput = "confidence";
constexpr auto kCacheOutput = "cache_out";

}  // namespace

PestoPitchDetector::PestoPitchDetector(const std::filesystem::path& modelPath, int sampleRate,
                                       ChannelFormat channelFormat, int blockSize,
                                       float confidenceThreshold)
    : _blockSize(blockSize),
      _numChannels(numChannels(channelFormat)),
      _confidenceThreshold(confidenceThreshold),
      _session(makeSession(modelPath)),
      _memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      _monoBuffer(blockSize) {
    // The cache input is (batch, cacheSize); get cacheSize by introspection.
    const auto cacheShape =
        _session.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    if (cacheShape.size() != 2 || cacheShape[1] <= 0) {
        throw std::runtime_error("Unexpected PESTO cache input shape in " + modelPath.string());
    }
    _cache.assign(static_cast<size_t>(cacheShape[1]), 0.f);
    (void)sampleRate;  // frozen into the model at export time; kept for clarity at call site
}

float PestoPitchDetector::process(const float* input, DebugOutput* debugOutput,
                                  std::vector<float>* debugOutputSignal) {
    (void)debugOutputSignal;  // no preprocessed signal to expose

    if (_numChannels == 1) {
        std::copy(input, input + _blockSize, _monoBuffer.begin());
    } else {
        for (auto i = 0; i < _blockSize; ++i) {
            _monoBuffer[i] = (input[2 * i] + input[2 * i + 1]) * 0.5f;
        }
    }

    const std::array<int64_t, 2> audioShape{1, static_cast<int64_t>(_blockSize)};
    const std::array<int64_t, 2> cacheShape{1, static_cast<int64_t>(_cache.size())};
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(_memoryInfo, _monoBuffer.data(), _monoBuffer.size(),
                                        audioShape.data(), audioShape.size()),
        Ort::Value::CreateTensor<float>(_memoryInfo, _cache.data(), _cache.size(),
                                        cacheShape.data(), cacheShape.size())};

    constexpr std::array<const char*, 2> inputNames{kAudioInput, kCacheInput};
    constexpr std::array<const char*, 3> outputNames{kPredictionOutput, kConfidenceOutput,
                                                     kCacheOutput};
    const auto outputs = _session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(),
                                      inputs.size(), outputNames.data(), outputNames.size());

    // prediction/confidence are (batch, timeSteps); with one chunk per call we
    // expect a single time step, but take the last one to be safe.
    const auto numPredictions =
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float prediction =
        numPredictions > 0 ? outputs[0].GetTensorData<float>()[numPredictions - 1] : 0.f;
    const auto numConfidences = outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
    const float confidence =
        numConfidences > 0 ? outputs[1].GetTensorData<float>()[numConfidences - 1] : 0.f;

    const auto cacheOutCount = outputs[2].GetTensorTypeAndShapeInfo().GetElementCount();
    assert(cacheOutCount == _cache.size());
    const auto* cacheOut = outputs[2].GetTensorData<float>();
    std::copy(cacheOut, cacheOut + std::min<size_t>(cacheOutCount, _cache.size()), _cache.begin());

    if (debugOutput) {
        (*debugOutput)["presenceScore"] = confidence;
    }

    if (std::isnan(prediction) || prediction <= 0.f || confidence < _confidenceThreshold) {
        return 0.f;
    }
    // prediction is in semitones (MIDI note number, A4 = 69 = 440 Hz).
    return 440.f * std::pow(2.f, (prediction - 69.f) / 12.f);
}

int PestoPitchDetector::delaySamples() const {
    // No FIFO buffering (chunk size == block size). The model's own analysis
    // window adds latency we don't know precisely; one block is a first guess
    // and a knob to tune if FPR/FNR look misaligned.
    return _blockSize;
}

}  // namespace saint
