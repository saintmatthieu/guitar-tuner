#pragma once

#include <string>
#include <vector>

#include "PitchDetector.h"
#include "PitchDetectorTypes.h"

// Forward-declare aubio's pitch handle so this header doesn't drag the C API
// into every translation unit that includes it. (fvec_t is an anonymous-struct
// typedef in aubio and can't be forward-declared, so the buffers it wraps live
// in std::vector members and the fvec_t views are built locally in the .cpp.)
extern "C" {
typedef struct _aubio_pitch_t aubio_pitch_t;
}

namespace saint {

// Benchmark-only wrapper around one of aubio's pitch-detection methods
// (https://github.com/aubio/aubio, GPL-3.0). Not part of the production library.
//
// aubio exposes several algorithms ("yin", "yinfft", "yinfast", "mcomb",
// "fcomb", "schmitt", "specacf"); the method is selected by string at
// construction. aubio keeps its own rolling analysis buffer of `bufSize`
// samples and accepts an arbitrary hop (== blockSize) per call, so each
// process() feeds exactly one block and no FIFO re-chunking is needed here.
class AubioPitchDetector : public PitchDetector {
   public:
    // bufSize is aubio's analysis window. It must be a power of two (the
    // default FFT backend requires it) and >= blockSize. Pass <= 0 to let the
    // wrapper pick a sensible default (next power of two >= 4 * blockSize,
    // floored at 2048).
    //
    // confidenceThreshold gates the output: process() returns 0 Hz when aubio's
    // confidence for the frame is below it. Pass < 0 (the default) to use the
    // method's built-in threshold (defaultConfidenceThreshold), calibrated to a
    // ~1% false-positive rate on the benchmark corpus.
    AubioPitchDetector(const std::string& method, int sampleRate, ChannelFormat channelFormat,
                       int blockSize, int bufSize = 0, float confidenceThreshold = -1.f);
    ~AubioPitchDetector() override;

    // The per-method confidence threshold at the ~1% false-positive-rate operating
    // point of the benchmark ROC. 0 for methods whose confidence carries no usable
    // voicing information (so they are effectively ungated).
    static float defaultConfidenceThreshold(const std::string& method);

    AubioPitchDetector(const AubioPitchDetector&) = delete;
    AubioPitchDetector& operator=(const AubioPitchDetector&) = delete;

    float process(const float* input, DebugOutput* debugOutput = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const int _blockSize;
    const int _numChannels;
    const int _bufSize;
    const float _confidenceThreshold;
    aubio_pitch_t* _pitch = nullptr;
    std::vector<float> _monoBuffer;  // one hop of (down-mixed) input, length blockSize
};

}  // namespace saint
