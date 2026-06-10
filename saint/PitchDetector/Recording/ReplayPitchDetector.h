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
 */
class ReplayPitchDetector : public PitchDetector {
   public:
    static std::unique_ptr<ReplayPitchDetector> fromFile(const std::filesystem::path&);

    /**
     * @brief Feeds the next stored block to the inner detector; `input` is
     * ignored. As with a live detector, the caller is expected to pass
     * `samplesPerBlockPerChannel * numChannels` samples per call (the size
     * stored in the file — see `config()`).
     *
     * @return the inner detector's estimate, or 0 when past the end of the
     * recording (see `numBlocksLeft()`).
     */
    float process(const float* input, DebugOutput* = nullptr,
                  std::vector<float>* debugOutputSignal = nullptr) override;
    int delaySamples() const override;

    const recording::PitchDetectorConfig& config() const;
    int numBlocks() const;
    int numBlocksLeft() const;

   private:
    explicit ReplayPitchDetector(recording::RecordingData);

    const recording::RecordingData _data;
    const std::unique_ptr<PitchDetector> _inner;
    const int _samplesPerBlock;
    int _blockIndex = 0;
};
}  // namespace saint
