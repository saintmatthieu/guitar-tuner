#include "PitchDetectorRecording.h"

#include <cstdint>
#include <fstream>

namespace saint {
namespace recording {
namespace {
constexpr uint16_t ieeeFloatFormatTag = 3;
constexpr uint16_t bitsPerSample = 32;

std::string tuningToString(Tuning tuning) {
    switch (tuning) {
        case Tuning::Standard:
            return "Standard";
    }
    return "Standard";
}

void writeU16(std::ostream& stream, uint16_t value) {
    const char bytes[2] = {static_cast<char>(value & 0xff), static_cast<char>(value >> 8 & 0xff)};
    stream.write(bytes, 2);
}

void writeU32(std::ostream& stream, uint32_t value) {
    const char bytes[4] = {static_cast<char>(value & 0xff), static_cast<char>(value >> 8 & 0xff),
                           static_cast<char>(value >> 16 & 0xff),
                           static_cast<char>(value >> 24 & 0xff)};
    stream.write(bytes, 4);
}

// Sub-chunk payloads must be padded to an even byte count.
uint32_t padded(uint32_t size) {
    return size + (size & 1);
}
}  // namespace

std::string serializeConfig(const PitchDetectorConfig& config) {
    return std::string(sampleRateKey) + "=" + std::to_string(config.sampleRate) + ";" +
           channelFormatKey + "=" +
           (config.channelFormat == ChannelFormat::Mono ? "Mono" : "Stereo") + ";" +
           samplesPerBlockPerChannelKey + "=" + std::to_string(config.samplesPerBlockPerChannel) +
           ";" + tuningKey + "=" + tuningToString(config.tuning);
}

bool writeWavFile(const std::filesystem::path& path, const PitchDetectorConfig& config,
                  const float* interleaved, size_t numSamples) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    const auto channels = static_cast<uint16_t>(numChannels(config.channelFormat));
    const auto sampleRate = static_cast<uint32_t>(config.sampleRate);
    const auto blockAlign = static_cast<uint16_t>(channels * sizeof(float));
    const auto dataSize = static_cast<uint32_t>(numSamples * sizeof(float));

    auto comment = serializeConfig(config);
    comment.push_back('\0');
    const auto commentSize = static_cast<uint32_t>(comment.size());
    const auto listSize = 4 /*"INFO"*/ + 8 + padded(commentSize);

    const auto riffSize = 4 /*"WAVE"*/ + (8 + 16) /*fmt*/ + (8 + 4) /*fact*/ + (8 + listSize) +
                          (8 + padded(dataSize));

    stream.write("RIFF", 4);
    writeU32(stream, riffSize);
    stream.write("WAVE", 4);

    stream.write("fmt ", 4);
    writeU32(stream, 16);
    writeU16(stream, ieeeFloatFormatTag);
    writeU16(stream, channels);
    writeU32(stream, sampleRate);
    writeU32(stream, sampleRate * blockAlign);
    writeU16(stream, blockAlign);
    writeU16(stream, bitsPerSample);

    stream.write("fact", 4);
    writeU32(stream, 4);
    writeU32(stream, static_cast<uint32_t>(numSamples / channels));

    stream.write("LIST", 4);
    writeU32(stream, listSize);
    stream.write("INFO", 4);
    stream.write("ICMT", 4);
    writeU32(stream, commentSize);
    stream.write(comment.data(), commentSize);
    if (commentSize & 1) {
        stream.put('\0');
    }

    stream.write("data", 4);
    writeU32(stream, dataSize);
    stream.write(reinterpret_cast<const char*>(interleaved), dataSize);
    if (dataSize & 1) {
        stream.put('\0');
    }

    return stream.good();
}

bool writeWavFile(const std::filesystem::path& path, const RecordingData& data) {
    return writeWavFile(path, data.config, data.interleaved.data(), data.interleaved.size());
}
}  // namespace recording
}  // namespace saint
