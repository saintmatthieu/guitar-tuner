#include "PitchDetectorRecording.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace saint {
namespace recording {
namespace {
constexpr uint16_t pcmFormatTag = 1;
constexpr uint16_t ieeeFloatFormatTag = 3;
constexpr uint16_t waveFormatExtensible = 0xfffe;
constexpr uint16_t bitsPerSample = 32;

std::string tuningToString(Tuning tuning) {
    switch (tuning) {
        case Tuning::Standard:
            return "Standard";
    }
    return "Standard";
}

std::optional<Tuning> tuningFromString(const std::string& str) {
    if (str == "Standard") {
        return Tuning::Standard;
    }
    return std::nullopt;
}

std::optional<ChannelFormat> channelFormatFromString(const std::string& str) {
    if (str == "Mono") {
        return ChannelFormat::Mono;
    }
    if (str == "Stereo") {
        return ChannelFormat::Stereo;
    }
    return std::nullopt;
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

bool readU16(std::istream& stream, uint16_t& value) {
    unsigned char bytes[2];
    if (!stream.read(reinterpret_cast<char*>(bytes), 2)) {
        return false;
    }
    value = static_cast<uint16_t>(bytes[0] | bytes[1] << 8);
    return true;
}

bool readU32(std::istream& stream, uint32_t& value) {
    unsigned char bytes[4];
    if (!stream.read(reinterpret_cast<char*>(bytes), 4)) {
        return false;
    }
    value = static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
            static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
    return true;
}

bool readFourCC(std::istream& stream, char (&id)[5]) {
    if (!stream.read(id, 4)) {
        return false;
    }
    id[4] = '\0';
    return true;
}

// Sub-chunk payloads must be padded to an even byte count.
uint32_t padded(uint32_t size) {
    return size + (size & 1);
}

// Converts a raw little-endian sample buffer into normalised [-1, 1] floats. Supports the
// formats a "standard" WAV file is likely to use - 8/16/24/32-bit integer PCM and 32/64-bit
// IEEE float - so a recording made outside the app can still be replayed. Returns false (leaving
// `out` unspecified) for any format we cannot interpret. Assumes a little-endian host, as does
// the rest of this file.
bool decodeSamples(const std::vector<unsigned char>& raw, uint16_t formatTag, uint16_t bits,
                   std::vector<float>& out) {
    const auto bytesPerSample = bits / 8u;
    if (bytesPerSample == 0) {
        return false;
    }
    const auto numSamples = raw.size() / bytesPerSample;
    out.resize(numSamples);

    if (formatTag == ieeeFloatFormatTag && bits == 32) {
        for (size_t i = 0; i < numSamples; ++i) {
            std::memcpy(&out[i], raw.data() + i * 4, 4);
        }
        return true;
    }
    if (formatTag == ieeeFloatFormatTag && bits == 64) {
        for (size_t i = 0; i < numSamples; ++i) {
            double value = 0;
            std::memcpy(&value, raw.data() + i * 8, 8);
            out[i] = static_cast<float>(value);
        }
        return true;
    }
    if (formatTag == pcmFormatTag && bits == 8) {
        // 8-bit PCM is unsigned with a bias of 128.
        for (size_t i = 0; i < numSamples; ++i) {
            out[i] = (static_cast<int>(raw[i]) - 128) / 128.0f;
        }
        return true;
    }
    if (formatTag == pcmFormatTag && bits == 16) {
        for (size_t i = 0; i < numSamples; ++i) {
            const auto value = static_cast<int16_t>(raw[i * 2] | raw[i * 2 + 1] << 8);
            out[i] = value / 32768.0f;
        }
        return true;
    }
    if (formatTag == pcmFormatTag && bits == 24) {
        for (size_t i = 0; i < numSamples; ++i) {
            int32_t value = raw[i * 3] | raw[i * 3 + 1] << 8 | raw[i * 3 + 2] << 16;
            if (value & 0x800000) {
                value -= 0x1000000;  // sign-extend the 24-bit value
            }
            out[i] = value / 8388608.0f;
        }
        return true;
    }
    if (formatTag == pcmFormatTag && bits == 32) {
        for (size_t i = 0; i < numSamples; ++i) {
            const auto u = static_cast<uint32_t>(raw[i * 4]) | static_cast<uint32_t>(raw[i * 4 + 1])
                                                                   << 8 |
                           static_cast<uint32_t>(raw[i * 4 + 2]) << 16 |
                           static_cast<uint32_t>(raw[i * 4 + 3]) << 24;
            out[i] = static_cast<float>(static_cast<int32_t>(u) / 2147483648.0);
        }
        return true;
    }
    return false;
}
}  // namespace

std::string serializeConfig(const PitchDetectorConfig& config) {
    return std::string(sampleRateKey) + "=" + std::to_string(config.sampleRate) + ";" +
           channelFormatKey + "=" +
           (config.channelFormat == ChannelFormat::Mono ? "Mono" : "Stereo") + ";" +
           samplesPerBlockPerChannelKey + "=" + std::to_string(config.samplesPerBlockPerChannel) +
           ";" + tuningKey + "=" + tuningToString(config.tuning);
}

std::optional<PitchDetectorConfig> deserializeConfig(const std::string& serialized) {
    std::unordered_map<std::string, std::string> entries;
    size_t pos = 0;
    while (pos < serialized.size()) {
        auto end = serialized.find(';', pos);
        if (end == std::string::npos) {
            end = serialized.size();
        }
        const auto pair = serialized.substr(pos, end - pos);
        const auto eq = pair.find('=');
        if (eq != std::string::npos) {
            entries[pair.substr(0, eq)] = pair.substr(eq + 1);
        }
        pos = end + 1;
    }

    if (entries.count(sampleRateKey) == 0 || entries.count(channelFormatKey) == 0 ||
        entries.count(samplesPerBlockPerChannelKey) == 0 || entries.count(tuningKey) == 0) {
        return std::nullopt;
    }

    const auto channelFormat = channelFormatFromString(entries[channelFormatKey]);
    const auto tuning = tuningFromString(entries[tuningKey]);
    if (!channelFormat.has_value() || !tuning.has_value()) {
        return std::nullopt;
    }

    try {
        return PitchDetectorConfig{std::stoi(entries[sampleRateKey]), *channelFormat,
                                   std::stoi(entries[samplesPerBlockPerChannelKey]), *tuning};
    } catch (const std::exception&) {
        return std::nullopt;
    }
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

std::optional<RecordingData> readWavFile(const std::filesystem::path& path, std::string* warning) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return std::nullopt;
    }

    char id[5];
    uint32_t riffSize = 0;
    if (!readFourCC(stream, id) || std::strcmp(id, "RIFF") != 0 || !readU32(stream, riffSize) ||
        !readFourCC(stream, id) || std::strcmp(id, "WAVE") != 0) {
        return std::nullopt;
    }

    std::optional<uint16_t> formatChannels;
    std::optional<uint32_t> formatSampleRate;
    uint16_t formatTag = 0;
    uint16_t formatBits = 0;
    std::optional<std::string> comment;
    std::vector<unsigned char> rawData;
    bool dataRead = false;

    while (readFourCC(stream, id)) {
        uint32_t chunkSize = 0;
        if (!readU32(stream, chunkSize)) {
            return std::nullopt;
        }
        const auto nextChunk = static_cast<std::streamoff>(stream.tellg()) + padded(chunkSize);
        if (std::strcmp(id, "fmt ") == 0 && chunkSize >= 16) {
            uint16_t tag = 0;
            uint16_t channels = 0;
            uint32_t sampleRate = 0;
            uint32_t byteRate = 0;
            uint16_t blockAlign = 0;
            uint16_t bits = 0;
            if (!readU16(stream, tag) || !readU16(stream, channels) ||
                !readU32(stream, sampleRate) || !readU32(stream, byteRate) ||
                !readU16(stream, blockAlign) || !readU16(stream, bits)) {
                return std::nullopt;
            }
            if (tag == waveFormatExtensible && chunkSize >= 40) {
                // The real format tag lives in the first two bytes of the SubFormat GUID.
                uint16_t cbSize = 0;
                uint16_t validBits = 0;
                uint32_t channelMask = 0;
                uint16_t subFormatTag = 0;
                if (!readU16(stream, cbSize) || !readU16(stream, validBits) ||
                    !readU32(stream, channelMask) || !readU16(stream, subFormatTag)) {
                    return std::nullopt;
                }
                tag = subFormatTag;
            }
            formatTag = tag;
            formatBits = bits;
            formatChannels = channels;
            formatSampleRate = sampleRate;
        } else if (std::strcmp(id, "LIST") == 0 && chunkSize >= 4) {
            char listType[5];
            if (!readFourCC(stream, listType)) {
                return std::nullopt;
            }
            if (std::strcmp(listType, "INFO") == 0) {
                auto remaining = chunkSize - 4;
                while (remaining >= 8) {
                    char subId[5];
                    uint32_t subSize = 0;
                    if (!readFourCC(stream, subId) || !readU32(stream, subSize)) {
                        return std::nullopt;
                    }
                    remaining -= 8;
                    if (subSize > remaining) {
                        break;
                    }
                    std::string value(subSize, '\0');
                    if (!stream.read(value.data(), subSize)) {
                        return std::nullopt;
                    }
                    if (padded(subSize) > subSize) {
                        stream.ignore(1);
                    }
                    remaining -= padded(subSize);
                    if (std::strcmp(subId, "ICMT") == 0) {
                        // Drop the null terminator (and anything after it).
                        comment = value.substr(0, value.find('\0'));
                    }
                }
            }
        } else if (std::strcmp(id, "data") == 0) {
            rawData.resize(chunkSize);
            if (!stream.read(reinterpret_cast<char*>(rawData.data()), chunkSize)) {
                return std::nullopt;
            }
            dataRead = true;
        }
        stream.clear();
        if (!stream.seekg(nextChunk)) {
            break;
        }
    }

    if (!dataRead || !formatChannels.has_value() || !formatSampleRate.has_value()) {
        return std::nullopt;
    }

    std::vector<float> interleaved;
    if (!decodeSamples(rawData, formatTag, formatBits, interleaved)) {
        return std::nullopt;
    }

    // A native app recording is 32-bit float carrying a config in its LIST INFO chunk that is
    // consistent with the format header. Anything else readable is treated as a foreign WAV:
    // converted above, replayed with a standard config, and flagged via `warning`.
    const bool isFloat32 = formatTag == ieeeFloatFormatTag && formatBits == bitsPerSample;
    std::optional<PitchDetectorConfig> config;
    if (isFloat32 && comment.has_value()) {
        if (auto parsed = deserializeConfig(*comment);
            parsed.has_value() && parsed->sampleRate == static_cast<int>(*formatSampleRate) &&
            numChannels(parsed->channelFormat) == static_cast<int>(*formatChannels)) {
            config = parsed;
        }
    }

    if (!config.has_value()) {
        if (*formatChannels != 1 && *formatChannels != 2) {
            // Only mono/stereo can be represented by a PitchDetectorConfig.
            return std::nullopt;
        }
        if (warning != nullptr) {
            *warning =
                "'" + path.string() +
                "' does not look like a guitar-tuner issue recording (expected a 32-bit float "
                "WAV with an embedded config). Converting it and replaying with the standard "
                "config.";
        }
        config = PitchDetectorConfig{
            static_cast<int>(*formatSampleRate),
            *formatChannels == 1 ? ChannelFormat::Mono : ChannelFormat::Stereo,
            defaultSamplesPerBlockPerChannel, Tuning::Standard};
    }

    // Only whole blocks can be replayed.
    const auto samplesPerBlock =
        config->samplesPerBlockPerChannel * numChannels(config->channelFormat);
    interleaved.resize(interleaved.size() / samplesPerBlock * samplesPerBlock);

    return RecordingData{*config, std::move(interleaved)};
}
}  // namespace recording
}  // namespace saint
