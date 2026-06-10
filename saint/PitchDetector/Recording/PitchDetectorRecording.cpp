#include "PitchDetectorRecording.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <unordered_map>

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

std::optional<RecordingData> readWavFile(const std::filesystem::path& path) {
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
    bool isFloat32 = false;
    std::optional<std::string> comment;
    std::vector<float> interleaved;
    bool dataRead = false;

    while (readFourCC(stream, id)) {
        uint32_t chunkSize = 0;
        if (!readU32(stream, chunkSize)) {
            return std::nullopt;
        }
        const auto nextChunk = static_cast<std::streamoff>(stream.tellg()) + padded(chunkSize);
        if (std::strcmp(id, "fmt ") == 0 && chunkSize >= 16) {
            uint16_t formatTag = 0;
            uint16_t channels = 0;
            uint32_t sampleRate = 0;
            uint32_t byteRate = 0;
            uint16_t blockAlign = 0;
            uint16_t bits = 0;
            if (!readU16(stream, formatTag) || !readU16(stream, channels) ||
                !readU32(stream, sampleRate) || !readU32(stream, byteRate) ||
                !readU16(stream, blockAlign) || !readU16(stream, bits)) {
                return std::nullopt;
            }
            isFloat32 = formatTag == ieeeFloatFormatTag && bits == bitsPerSample;
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
            interleaved.resize(chunkSize / sizeof(float));
            if (!stream.read(reinterpret_cast<char*>(interleaved.data()),
                             interleaved.size() * sizeof(float))) {
                return std::nullopt;
            }
            dataRead = true;
        }
        stream.clear();
        if (!stream.seekg(nextChunk)) {
            break;
        }
    }

    if (!isFloat32 || !dataRead || !comment.has_value()) {
        return std::nullopt;
    }
    const auto config = deserializeConfig(*comment);
    if (!config.has_value() ||
        config->sampleRate != static_cast<int>(formatSampleRate.value_or(0)) ||
        numChannels(config->channelFormat) != static_cast<int>(formatChannels.value_or(0))) {
        return std::nullopt;
    }

    // Only whole blocks can be replayed.
    const auto samplesPerBlock =
        config->samplesPerBlockPerChannel * numChannels(config->channelFormat);
    interleaved.resize(interleaved.size() / samplesPerBlock * samplesPerBlock);

    return RecordingData{*config, std::move(interleaved)};
}
}  // namespace recording
}  // namespace saint
