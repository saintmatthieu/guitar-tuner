#include "CoreAudioInput.h"

#include <iostream>

namespace saint {

CoreAudioInput::CoreAudioInput(int sampleRate, int blockSize, const std::string& device)
    : _sampleRate(sampleRate), _blockSize(blockSize) {
    if (device != "default") {
        std::cerr << "Warning: device selection is not supported on macOS, using default input"
                  << std::endl;
    }
    _accumulator.reserve(2 * blockSize);
}

CoreAudioInput::~CoreAudioInput() {
    stop();
}

void CoreAudioInput::queueCallback(void* userData, AudioQueueRef queue, AudioQueueBufferRef buffer,
                                   const AudioTimeStamp*, UInt32,
                                   const AudioStreamPacketDescription*) {
    static_cast<CoreAudioInput*>(userData)->handleBuffer(queue, buffer);
}

void CoreAudioInput::handleBuffer(AudioQueueRef queue, AudioQueueBufferRef buffer) {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_running) {
            return;
        }
    }

    const auto* samples = static_cast<const float*>(buffer->mAudioData);
    const int numSamples = static_cast<int>(buffer->mAudioDataByteSize / sizeof(float));
    _accumulator.insert(_accumulator.end(), samples, samples + numSamples);

    size_t offset = 0;
    while (_accumulator.size() - offset >= static_cast<size_t>(_blockSize)) {
        if (_callback) {
            _callback(_accumulator.data() + offset, _blockSize);
        }
        offset += _blockSize;
    }
    _accumulator.erase(_accumulator.begin(), _accumulator.begin() + offset);

    AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr);
}

bool CoreAudioInput::start(AudioCallback callback) {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_running) {
            return false;
        }
        _running = true;
    }

    _callback = std::move(callback);

    AudioStreamBasicDescription format{};
    format.mSampleRate = _sampleRate;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked;
    format.mChannelsPerFrame = 1;
    format.mBitsPerChannel = 32;
    format.mBytesPerFrame = sizeof(float);
    format.mFramesPerPacket = 1;
    format.mBytesPerPacket = sizeof(float);

    OSStatus err =
        AudioQueueNewInput(&format, queueCallback, this, nullptr, nullptr, 0, &_queue);
    if (err != noErr) {
        std::cerr << "Cannot open audio input queue (OSStatus " << err << ")" << std::endl;
        std::lock_guard<std::mutex> lock(_mutex);
        _running = false;
        return false;
    }

    constexpr int kNumBuffers = 3;
    const UInt32 bufferByteSize = static_cast<UInt32>(_blockSize) * sizeof(float);
    for (int i = 0; i < kNumBuffers; ++i) {
        AudioQueueBufferRef buffer;
        err = AudioQueueAllocateBuffer(_queue, bufferByteSize, &buffer);
        if (err == noErr) {
            err = AudioQueueEnqueueBuffer(_queue, buffer, 0, nullptr);
        }
        if (err != noErr) {
            std::cerr << "Cannot allocate audio buffer (OSStatus " << err << ")" << std::endl;
            AudioQueueDispose(_queue, true);
            _queue = nullptr;
            std::lock_guard<std::mutex> lock(_mutex);
            _running = false;
            return false;
        }
    }

    err = AudioQueueStart(_queue, nullptr);
    if (err != noErr) {
        std::cerr << "Cannot start audio capture (OSStatus " << err << ")" << std::endl;
        AudioQueueDispose(_queue, true);
        _queue = nullptr;
        std::lock_guard<std::mutex> lock(_mutex);
        _running = false;
        return false;
    }

    // Match the ALSA implementation: block until stop() is called (possibly
    // from within the audio callback). The queue runs on its own thread.
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _stopped.wait(lock, [this] { return !_running; });
    }

    AudioQueueStop(_queue, true);
    AudioQueueDispose(_queue, true);
    _queue = nullptr;

    return true;
}

void CoreAudioInput::stop() {
    std::lock_guard<std::mutex> lock(_mutex);
    _running = false;
    _stopped.notify_all();
}

}  // namespace saint
