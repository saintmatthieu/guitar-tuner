#include "PyinPitchDetector.h"

#include <algorithm>
#include <cmath>

#include "MonoPitchHMM.h"
#include "Yin.h"

namespace saint {

PyinPitchDetector::PyinPitchDetector(int sampleRate, ChannelFormat channelFormat, int blockSize,
                                     int frameSize, int fixedLag, int threshDistr, float lowAmp)
    : _blockSize(blockSize),
      _numChannels(numChannels(channelFormat)),
      _frameSize(frameSize),
      _fixedLag(std::max(1, fixedLag)),
      _lowAmp(lowAmp),
      _yin(std::make_unique<Yin>(static_cast<size_t>(frameSize), static_cast<size_t>(sampleRate),
                                 0.0, true)),
      _hmm(std::make_unique<MonoPitchHMM>(_fixedLag)),
      _window(static_cast<size_t>(frameSize), 0.0) {
    // pYIN itself uses the threshold *distribution* rather than a single
    // threshold; the 0.0 single-threshold above is irrelevant to
    // processProbabilisticYin().
    _yin->setThresholdDistr(static_cast<float>(threshDistr));
    // MonoPitchHMM's constructor already calls build().
}

PyinPitchDetector::~PyinPitchDetector() = default;

PitchDetectionResult PyinPitchDetector::process(const float* input, DebugOutput* debugOutput,
                                                std::vector<float>* /*debugOutputSignal*/) {
    // 1) Slide the rolling analysis window left by one block and append this
    //    block's (down-mixed) mono samples at the tail.
    std::move(_window.begin() + _blockSize, _window.end(), _window.begin());
    double* tail = _window.data() + (_frameSize - _blockSize);
    double blockSumSq = 0.0;
    if (_numChannels == 1) {
        for (int i = 0; i < _blockSize; ++i) {
            tail[i] = input[i];
            blockSumSq += tail[i] * tail[i];
        }
    } else {
        for (int i = 0; i < _blockSize; ++i) {
            tail[i] = 0.5 * (static_cast<double>(input[2 * i]) + input[2 * i + 1]);
            blockSumSq += tail[i] * tail[i];
        }
    }
    const double blockRms = std::sqrt(blockSumSq / _blockSize);
    const bool isLowAmplitude = blockRms < _lowAmp;

    // 2) One probabilistic-YIN frame over the trailing window.
    const Yin::YinOutput yo = _yin->processProbabilisticYin(_window.data());

    // 3) Convert each YIN candidate from Hz to a MIDI note number (the form
    //    MonoPitchHMM expects), suppressing candidate weight in low-amplitude
    //    blocks, exactly as the pYIN reference (PYinVamp) does.
    const float lowAmpFactor =
        isLowAmplitude ? (blockRms + 0.01f * _lowAmp) / (1.01f * _lowAmp) : 1.f;
    std::vector<std::pair<double, double>> midiProb;
    midiProb.reserve(yo.freqProb.size());
    float voicedProb = 0.f;
    for (const auto& fp : yo.freqProb) {
        if (fp.first <= 0.0) {
            continue;
        }
        const double midi = 12.0 * std::log2(fp.first / 440.0) + 69.0;
        midiProb.emplace_back(midi, fp.second * lowAmpFactor);
        voicedProb += static_cast<float>(fp.second);
    }
    voicedProb = std::clamp(voicedProb, 0.f, 1.f);

    // 4) Feed the frame's candidate distribution into the HMM, advancing the
    //    fixed-lag Viterbi trellis by one frame.
    const std::vector<double> obs = _hmm->calculateObsProb(midiProb);
    if (!_initialised) {
        _hmm->initialise(obs);
        _initialised = true;
    } else {
        _hmm->process(obs);
    }
    // Keep our per-frame history trimmed exactly like SparseHMM trims its trellis
    // (m_psi), so history.front() corresponds to track()'s oldest path entry.
    _midiProbHistory.push_back(std::move(midiProb));
    _voicedProbHistory.push_back(voicedProb);
    while (static_cast<int>(_midiProbHistory.size()) > _fixedLag) {
        _midiProbHistory.pop_front();
        _voicedProbHistory.pop_front();
    }

    // 5) Once the lag has filled, emit the smoothed estimate for the oldest
    //    retained frame (decoded with fixedLag-1 frames of future context). Before
    //    that we are still warming up and emit nothing.
    float frequency = 0.f;
    float presence = 0.f;
    if (static_cast<int>(_midiProbHistory.size()) >= _fixedLag) {
        const std::vector<int> path = _hmm->track();
        if (!path.empty()) {
            // nearestFreq maps the HMM state back to one of the frame's candidate
            // frequencies (in Hz); it returns a negative value for unvoiced states.
            const float smoothed = _hmm->nearestFreq(path.front(), _midiProbHistory.front());
            frequency = smoothed > 0.f ? smoothed : 0.f;
        }
        presence = _voicedProbHistory.front();
    }

    if (debugOutput) {
        (*debugOutput)["presenceScore"] = presence;
    }
    if (frequency == 0.f) {
        return {};
    }
    return {frequency, PitchBucket::inRange};
}

int PyinPitchDetector::delaySamples() const {
    // The emitted estimate is the frame (fixedLag-1) blocks behind the current
    // one; that frame's analysis window is centred frameSize/2 before its end.
    return _frameSize / 2 + (_fixedLag - 1) * _blockSize;
}
}  // namespace saint
