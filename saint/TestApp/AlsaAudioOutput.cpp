#include "AlsaAudioOutput.h"

#include <iostream>

namespace saint {

AlsaAudioOutput::AlsaAudioOutput(int sampleRate, int channels, int blockSize,
                                 const std::string& device)
    : _sampleRate(sampleRate), _channels(channels), _blockSize(blockSize), _device(device) {}

AlsaAudioOutput::~AlsaAudioOutput() {
    close();
}

bool AlsaAudioOutput::open() {
    int err;

    // Open PCM device for playback
    err = snd_pcm_open(&_pcmHandle, _device.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        std::cerr << "Cannot open playback device " << _device << ": " << snd_strerror(err)
                  << std::endl;
        return false;
    }

    // Set hardware parameters
    snd_pcm_hw_params_t* hwParams;
    snd_pcm_hw_params_alloca(&hwParams);

    err = snd_pcm_hw_params_any(_pcmHandle, hwParams);
    if (err < 0) {
        std::cerr << "Cannot initialize playback parameters: " << snd_strerror(err) << std::endl;
        close();
        return false;
    }

    // Set access type
    err = snd_pcm_hw_params_set_access(_pcmHandle, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        std::cerr << "Cannot set playback access type: " << snd_strerror(err) << std::endl;
        close();
        return false;
    }

    // The recording is held as 32-bit float, so play it back as such and let the
    // "default" device's plug layer convert to whatever the hardware wants.
    err = snd_pcm_hw_params_set_format(_pcmHandle, hwParams, SND_PCM_FORMAT_FLOAT_LE);
    if (err < 0) {
        std::cerr << "Cannot set playback sample format: " << snd_strerror(err) << std::endl;
        close();
        return false;
    }

    // Set channel count (matches the recording: mono or stereo)
    err = snd_pcm_hw_params_set_channels(_pcmHandle, hwParams,
                                         static_cast<unsigned int>(_channels));
    if (err < 0) {
        std::cerr << "Cannot set playback channel count (" << _channels
                  << "): " << snd_strerror(err) << std::endl;
        close();
        return false;
    }

    // Set sample rate
    unsigned int actualRate = static_cast<unsigned int>(_sampleRate);
    err = snd_pcm_hw_params_set_rate_near(_pcmHandle, hwParams, &actualRate, nullptr);
    if (err < 0) {
        std::cerr << "Cannot set playback sample rate: " << snd_strerror(err) << std::endl;
        close();
        return false;
    }
    if (actualRate != static_cast<unsigned int>(_sampleRate)) {
        std::cerr << "Warning: Requested playback rate " << _sampleRate << " Hz, got " << actualRate
                  << " Hz" << std::endl;
    }

    // Period size (one replay block)
    snd_pcm_uframes_t periodSize = static_cast<snd_pcm_uframes_t>(_blockSize);
    snd_pcm_hw_params_set_period_size_near(_pcmHandle, hwParams, &periodSize, nullptr);

    // A few periods of buffering keeps writes from underrunning while staying
    // close enough to real time that the audio tracks the on-screen tuner.
    snd_pcm_uframes_t bufferSize = static_cast<snd_pcm_uframes_t>(_blockSize) * 4;
    snd_pcm_hw_params_set_buffer_size_near(_pcmHandle, hwParams, &bufferSize);

    // Apply hardware parameters
    err = snd_pcm_hw_params(_pcmHandle, hwParams);
    if (err < 0) {
        std::cerr << "Cannot set playback hardware parameters: " << snd_strerror(err) << std::endl;
        close();
        return false;
    }

    // Prepare the device
    err = snd_pcm_prepare(_pcmHandle);
    if (err < 0) {
        std::cerr << "Cannot prepare playback device: " << snd_strerror(err) << std::endl;
        close();
        return false;
    }

    return true;
}

void AlsaAudioOutput::close() {
    if (_pcmHandle) {
        snd_pcm_drain(_pcmHandle);
        snd_pcm_close(_pcmHandle);
        _pcmHandle = nullptr;
    }
}

bool AlsaAudioOutput::write(const float* interleaved, int framesPerChannel) {
    const float* ptr = interleaved;
    snd_pcm_uframes_t remaining = static_cast<snd_pcm_uframes_t>(framesPerChannel);

    while (remaining > 0) {
        snd_pcm_sframes_t written = snd_pcm_writei(_pcmHandle, ptr, remaining);

        if (written < 0) {
            // Recover from underruns/suspends; the second argument silences the
            // recovery's own diagnostics.
            written = snd_pcm_recover(_pcmHandle, static_cast<int>(written), 1);
            if (written < 0) {
                std::cerr << "Audio playback error: " << snd_strerror(static_cast<int>(written))
                          << std::endl;
                return false;
            }
            continue;
        }

        ptr += static_cast<size_t>(written) * _channels;
        remaining -= static_cast<snd_pcm_uframes_t>(written);
    }

    return true;
}

}  // namespace saint
