#pragma once

#include <filesystem>
#include <memory>

#include "PitchDetector.h"
#include "Recording/PitchDetectorRecording.h"

namespace saint {
/**
 * @brief Decorator that replays a session saved by `RecordingPitchDetector`:
 * it reads the recorded audio and config from the WAV file and feeds the
 * stored blocks to a fresh `PitchDetector` created via
 * `PitchDetectorFactory::createInstance()`.
 *
 * Like `readWavFile`, this is offline debug/test tooling (used by ReplayApp and
 * the recording tests), not part of the production PitchDetector library.
 */
class ReplayPitchDetector : public PitchDetector {
   public:
    // If `warning` is non-null and the file isn't a native app recording, it is set to an
    // explanatory message (the file is still loaded with a standard config; see `readWavFile`).
    // `lowBand` configures below-range detection, which the recording does not carry.
    static std::unique_ptr<ReplayPitchDetector> fromFile(const std::filesystem::path&,
                                                         std::string* warning = nullptr,
                                                         LowBandConfig lowBand = {});

    /**
     * @brief Feeds the next stored block to the inner detector; `input` is
     * ignored. As with a live detector, the caller is expected to pass
     * `samplesPerBlockPerChannel * numChannels` samples per call (the size
     * stored in the file — see `config()`).
     *
     * @return the inner detector's estimate, or 0 when past the end of the
     * recording (see `numBlocksLeft()`).
     */
    PitchDetectionResult process(const float* input, DebugOutput* = nullptr,
                                 std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;
    std::pair<float, float> pitchSearchRange() const override;

    const recording::PitchDetectorConfig& config() const;
    int numBlocks() const;
    int numBlocksLeft() const;

    /**
     * @brief Pointer to the interleaved samples of the block the next
     * `process()` call will consume, or `nullptr` once past the end. The block
     * spans `config().samplesPerBlockPerChannel * numChannels(...)` samples.
     * Lets callers (e.g. the ReplayApp) play the recorded audio back as it is
     * replayed, without re-reading the file.
     */
    const float* peekBlock() const;

   private:
    ReplayPitchDetector(recording::RecordingData, LowBandConfig);

    const recording::RecordingData _data;
    const std::unique_ptr<PitchDetector> _inner;
    const int _samplesPerBlock;
    int _blockIndex = 0;
};
}  // namespace saint
