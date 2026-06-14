#include "AubioPitchDetector.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include <aubio.h>

namespace saint {
namespace {

// aubio's default FFT backend (OOURA) only accepts power-of-two window sizes,
// and the analysis window must hold at least one hop. Round 4 * blockSize up to
// the next power of two, floored at 2048, which keeps a useful window for the
// 10 ms blocks the benchmark feeds (e.g. 441 -> 2048).
int defaultBufSize(int blockSize) {
    int target = std::max(2048, 4 * blockSize);
    int pow2 = 1;
    while (pow2 < target) {
        pow2 <<= 1;
    }
    return pow2;
}

}  // namespace

float AubioPitchDetector::defaultConfidenceThreshold(const std::string& method) {
    // Confidence at the ~1% false-positive-rate operating point of each method's
    // benchmark ROC (eval/out/roc_curve_aubio-<method>.py, with
    // updateBenchmarkReferences). mcomb/fcomb/schmitt return a constant confidence
    // (aubio has no confidence callback for them), so their ROC is degenerate and
    // the operating point is 0 - i.e. effectively ungated.
    static const std::unordered_map<std::string, float> thresholds{
        {"yin", 0.957062f},  {"yinfft", 0.787948f}, {"yinfast", 0.957064f}, {"mcomb", 0.f},
        {"fcomb", 0.f},      {"schmitt", 0.f},      {"specacf", 0.85f},
    };
    const auto it = thresholds.find(method);
    return it != thresholds.end() ? it->second : 0.f;
}

AubioPitchDetector::AubioPitchDetector(const std::string& method, int sampleRate,
                                       ChannelFormat channelFormat, int blockSize, int bufSize,
                                       float confidenceThreshold)
    : _blockSize(blockSize),
      _numChannels(numChannels(channelFormat)),
      _bufSize(bufSize > 0 ? bufSize : defaultBufSize(blockSize)),
      _confidenceThreshold(confidenceThreshold >= 0.f ? confidenceThreshold
                                                      : defaultConfidenceThreshold(method)),
      _monoBuffer(static_cast<size_t>(blockSize)) {
    _pitch = new_aubio_pitch(method.c_str(), static_cast<uint_t>(_bufSize),
                             static_cast<uint_t>(_blockSize), static_cast<uint_t>(sampleRate));
    if (_pitch == nullptr) {
        throw std::runtime_error("Failed to create aubio pitch detector for method '" + method +
                                 "' (bufSize=" + std::to_string(_bufSize) +
                                 ", hopSize=" + std::to_string(_blockSize) + ")");
    }
    aubio_pitch_set_unit(_pitch, "Hz");
}

AubioPitchDetector::~AubioPitchDetector() {
    del_aubio_pitch(_pitch);
}

float AubioPitchDetector::process(const float* input, DebugOutput* debugOutput,
                                  std::vector<float>* debugOutputSignal) {
    (void)debugOutputSignal;  // no preprocessed signal to expose

    if (_numChannels == 1) {
        std::copy(input, input + _blockSize, _monoBuffer.begin());
    } else {
        for (auto i = 0; i < _blockSize; ++i) {
            _monoBuffer[i] = (input[2 * i] + input[2 * i + 1]) * 0.5f;
        }
    }

    // fvec_t is a thin {length, data} view; build views over our own storage so
    // we don't allocate aubio buffers per instance.
    fvec_t in{static_cast<uint_t>(_blockSize), _monoBuffer.data()};
    float outValue = 0.f;
    fvec_t out{1, &outValue};

    aubio_pitch_do(_pitch, &in, &out);
    const float frequency = outValue;
    const float confidence = aubio_pitch_get_confidence(_pitch);

    if (debugOutput) {
        (*debugOutput)["presenceScore"] = confidence;
    }

    if (std::isnan(frequency) || frequency <= 0.f || confidence < _confidenceThreshold) {
        return 0.f;
    }
    return frequency;
}

int AubioPitchDetector::delaySamples() const {
    // The estimate is computed over the trailing analysis window, so it best
    // matches the centre of that window. Half the window is a first guess and a
    // knob to tune if FPR/FNR look misaligned against the ground truth.
    return _bufSize / 2;
}

}  // namespace saint
