#pragma once

#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "PitchDetector.h"
#include "PitchDetectorTypes.h"

// Forward-declare the vendored pYIN classes (global namespace, no Vamp SDK) so
// this header stays free of the upstream headers. The real includes live in the
// .cpp.
class Yin;
class MonoPitchHMM;

namespace saint {

// Benchmark-only wrapper around the pYIN (probabilistic YIN) reference algorithm
// by Mauch & Dixon (https://github.com/c4dm/pyin, GPL-2.0-or-later). Not part of
// the production library — see pyin-integration.md at the repo root.
//
// pYIN runs in two stages: a probabilistic-YIN front end emits, per analysis
// frame, a set of (frequency, probability) pitch candidates; an HMM then smooths
// these into a pitch track. The reference decodes that HMM offline (Viterbi over
// the whole file). To fit the streaming, per-block PitchDetector interface, we use
// the reference SparseHMM's built-in fixed-lag decoding: the estimate returned for
// the current block is the Viterbi-optimal pitch of the frame `fixedLag-1` blocks
// in the past, decoded with that many frames of future context. This trades a
// bounded latency (reported via delaySamples()) for online operation.
class PyinPitchDetector : public PitchDetector {
   public:
    // frameSize is pYIN's analysis window; it must be a power of two and >=
    // blockSize (default 2048, pYIN's reference value). fixedLag is the HMM
    // smoothing lag in frames (>= 1); the larger it is the more future context
    // each estimate sees, at the cost of (fixedLag-1) blocks of latency.
    // threshDistr selects the YIN threshold-distribution prior (0 uniform; 1..4
    // Beta with mean 0.10/0.15/0.20/0.30; default 2, pYIN's reference).
    PyinPitchDetector(int sampleRate, ChannelFormat channelFormat, int blockSize,
                      int frameSize = 2048, int fixedLag = 20, int threshDistr = 2,
                      float lowAmp = 0.1f);
    ~PyinPitchDetector() override;

    PyinPitchDetector(const PyinPitchDetector&) = delete;
    PyinPitchDetector& operator=(const PyinPitchDetector&) = delete;

    float process(const float* input, DebugOutput* debugOutput = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

   private:
    const int _blockSize;
    const int _numChannels;
    const int _frameSize;
    const int _fixedLag;
    const float _lowAmp;  // block-RMS below which YIN candidates are suppressed

    std::unique_ptr<Yin> _yin;
    std::unique_ptr<MonoPitchHMM> _hmm;

    // Trailing _frameSize mono samples; one pYIN frame is computed per block over
    // this window (hop == blockSize).
    std::vector<double> _window;

    // Per-frame pitch-candidate sets (as (MIDI note, probability) pairs, the form
    // MonoPitchHMM expects) and voiced-probability scores, retained in lockstep
    // with the HMM's fixed-lag trellis so the lag-delayed estimate can be mapped
    // back to a frequency (and a presence score).
    std::deque<std::vector<std::pair<double, double>>> _midiProbHistory;
    std::deque<float> _voicedProbHistory;

    bool _initialised = false;
};

}  // namespace saint
