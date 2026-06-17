#pragma once

#include <alsa/asoundlib.h>

#include <string>

namespace saint {

// Minimal blocking ALSA playback, used by the ReplayApp to render a recording's
// audio while it is replayed in real time. Each write() blocks until the device
// has room for the block, which is what paces the replay loop - so the caller
// needs no separate sleep when audio is playing.
class AlsaAudioOutput {
   public:
    AlsaAudioOutput(int sampleRate, int channels, int blockSize,
                    const std::string& device = "default");
    ~AlsaAudioOutput();

    // Non-copyable
    AlsaAudioOutput(const AlsaAudioOutput&) = delete;
    AlsaAudioOutput& operator=(const AlsaAudioOutput&) = delete;

    // Opens and configures the device for 32-bit float interleaved playback.
    // Returns false (after printing a diagnostic) if the device can't be opened
    // or configured; the caller can then carry on without audio.
    bool open();

    // Writes one block of `framesPerChannel * channels` interleaved float
    // samples, recovering from underruns. Returns false on an unrecoverable
    // error.
    bool write(const float* interleaved, int framesPerChannel);

   private:
    void close();

    const int _sampleRate;
    const int _channels;
    const int _blockSize;
    const std::string _device;

    snd_pcm_t* _pcmHandle = nullptr;
};

}  // namespace saint
