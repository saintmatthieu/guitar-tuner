#pragma once

#include <alsa/asoundlib.h>

#include <functional>
#include <string>
#include <vector>

namespace saint {

class AlsaAudioInput {
   public:
    using AudioCallback = std::function<void(const float* samples, int numSamples)>;

    AlsaAudioInput(int sampleRate, int blockSize, const std::string& device = "default");
    ~AlsaAudioInput();

    // Non-copyable
    AlsaAudioInput(const AlsaAudioInput&) = delete;
    AlsaAudioInput& operator=(const AlsaAudioInput&) = delete;

    bool start(AudioCallback callback);
    void stop();

    int sampleRate() const {
        return _sampleRate;
    }
    int blockSize() const {
        return _blockSize;
    }

   private:
    bool openDevice();
    void closeDevice();
    void captureLoop();

    const int _sampleRate;
    const int _blockSize;
    const std::string _device;

    snd_pcm_t* _pcmHandle = nullptr;
    AudioCallback _callback;
    bool _running = false;
    std::vector<int16_t> _buffer;
    std::vector<float> _floatBuffer;
};

}  // namespace saint
