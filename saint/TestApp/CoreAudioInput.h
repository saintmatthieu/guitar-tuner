#pragma once

#include <AudioToolbox/AudioToolbox.h>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace saint {

class CoreAudioInput {
   public:
    using AudioCallback = std::function<void(const float* samples, int numSamples)>;

    CoreAudioInput(int sampleRate, int blockSize, const std::string& device = "default");
    ~CoreAudioInput();

    // Non-copyable
    CoreAudioInput(const CoreAudioInput&) = delete;
    CoreAudioInput& operator=(const CoreAudioInput&) = delete;

    bool start(AudioCallback callback);
    void stop();

    int sampleRate() const {
        return _sampleRate;
    }
    int blockSize() const {
        return _blockSize;
    }

   private:
    static void queueCallback(void* userData, AudioQueueRef queue, AudioQueueBufferRef buffer,
                              const AudioTimeStamp* startTime, UInt32 numPackets,
                              const AudioStreamPacketDescription* packetDescs);
    void handleBuffer(AudioQueueRef queue, AudioQueueBufferRef buffer);

    const int _sampleRate;
    const int _blockSize;

    AudioQueueRef _queue = nullptr;
    AudioCallback _callback;
    bool _running = false;
    std::mutex _mutex;
    std::condition_variable _stopped;
    // Queue buffers may hold fewer or more frames than _blockSize; accumulate
    // here so the callback always receives exactly _blockSize samples.
    std::vector<float> _accumulator;
};

}  // namespace saint
