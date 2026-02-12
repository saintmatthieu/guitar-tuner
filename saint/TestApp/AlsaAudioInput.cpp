#include "AlsaAudioInput.h"

#include <iostream>
#include <limits>

namespace saint {

AlsaAudioInput::AlsaAudioInput(int sampleRate, int blockSize, const std::string& device)
    : _sampleRate(sampleRate),
      _blockSize(blockSize),
      _device(device),
      _buffer(blockSize),
      _floatBuffer(blockSize) {}

AlsaAudioInput::~AlsaAudioInput() {
    stop();
}

bool AlsaAudioInput::openDevice() {
    int err;

    // Open PCM device for recording
    err = snd_pcm_open(&_pcmHandle, _device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        std::cerr << "Cannot open audio device " << _device << ": " << snd_strerror(err)
                  << std::endl;
        return false;
    }

    // Set hardware parameters
    snd_pcm_hw_params_t* hwParams;
    snd_pcm_hw_params_alloca(&hwParams);

    err = snd_pcm_hw_params_any(_pcmHandle, hwParams);
    if (err < 0) {
        std::cerr << "Cannot initialize hardware parameters: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }

    // Set access type
    err = snd_pcm_hw_params_set_access(_pcmHandle, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        std::cerr << "Cannot set access type: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }

    // Set sample format (16-bit signed little-endian)
    err = snd_pcm_hw_params_set_format(_pcmHandle, hwParams, SND_PCM_FORMAT_S16_LE);
    if (err < 0) {
        std::cerr << "Cannot set sample format: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }

    // Set mono
    err = snd_pcm_hw_params_set_channels(_pcmHandle, hwParams, 1);
    if (err < 0) {
        std::cerr << "Cannot set channel count: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }

    // Set sample rate
    unsigned int actualRate = _sampleRate;
    err = snd_pcm_hw_params_set_rate_near(_pcmHandle, hwParams, &actualRate, nullptr);
    if (err < 0) {
        std::cerr << "Cannot set sample rate: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }
    if (actualRate != static_cast<unsigned int>(_sampleRate)) {
        std::cerr << "Warning: Requested sample rate " << _sampleRate << " Hz, got " << actualRate
                  << " Hz" << std::endl;
    }

    // Set period size (block size)
    snd_pcm_uframes_t periodSize = _blockSize;
    err = snd_pcm_hw_params_set_period_size_near(_pcmHandle, hwParams, &periodSize, nullptr);
    if (err < 0) {
        std::cerr << "Cannot set period size: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }

    // Apply hardware parameters
    err = snd_pcm_hw_params(_pcmHandle, hwParams);
    if (err < 0) {
        std::cerr << "Cannot set hardware parameters: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }

    // Prepare the device
    err = snd_pcm_prepare(_pcmHandle);
    if (err < 0) {
        std::cerr << "Cannot prepare audio device: " << snd_strerror(err) << std::endl;
        closeDevice();
        return false;
    }

    return true;
}

void AlsaAudioInput::closeDevice() {
    if (_pcmHandle) {
        snd_pcm_close(_pcmHandle);
        _pcmHandle = nullptr;
    }
}

bool AlsaAudioInput::start(AudioCallback callback) {
    if (_running) {
        return false;
    }

    _callback = std::move(callback);

    if (!openDevice()) {
        return false;
    }

    _running = true;
    captureLoop();

    return true;
}

void AlsaAudioInput::stop() {
    _running = false;
    closeDevice();
}

void AlsaAudioInput::captureLoop() {
    constexpr float kNormalizationFactor = 1.0f / std::numeric_limits<int16_t>::max();

    while (_running) {
        snd_pcm_sframes_t framesRead = snd_pcm_readi(_pcmHandle, _buffer.data(), _blockSize);

        if (framesRead < 0) {
            // Try to recover from errors
            framesRead = snd_pcm_recover(_pcmHandle, framesRead, 0);
            if (framesRead < 0) {
                std::cerr << "Audio read error: " << snd_strerror(framesRead) << std::endl;
                break;
            }
        }

        if (framesRead > 0) {
            // Convert int16_t to float
            for (int i = 0; i < framesRead; ++i) {
                _floatBuffer[i] = static_cast<float>(_buffer[i]) * kNormalizationFactor;
            }

            if (_callback) {
                _callback(_floatBuffer.data(), framesRead);
            }
        }
    }
}

}  // namespace saint
