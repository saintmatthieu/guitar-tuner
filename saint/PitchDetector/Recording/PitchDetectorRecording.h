#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "PitchDetectorTypes.h"

namespace saint {
namespace recording {

struct PitchDetectorConfig {
    int sampleRate;
    ChannelFormat channelFormat;
    int samplesPerBlockPerChannel;
    Tuning tuning;
};

struct RecordingData {
    PitchDetectorConfig config;
    std::vector<float> interleaved;
};

// The config is serialized as `key=value` pairs separated by `;` into the ICMT
// (comment) sub-chunk of the WAV file's LIST INFO chunk, so the file stays
// playable in any audio tool while carrying everything needed for replay.
constexpr auto sampleRateKey = "sampleRate";
constexpr auto channelFormatKey = "channelFormat";
constexpr auto samplesPerBlockPerChannelKey = "samplesPerBlockPerChannel";
constexpr auto tuningKey = "tuning";

std::string serializeConfig(const PitchDetectorConfig&);
std::optional<PitchDetectorConfig> deserializeConfig(const std::string&);

// Writes a 32-bit float WAV file with the config in the LIST INFO chunk. Clients receiving a
// `RecordingData` (see `IssueReportingPitchDetector::startIssueRecording`) must persist it with
// this function - and not a generic WAV writer - so that the metadata needed by
// `ReplayPitchDetector` is included.
bool writeWavFile(const std::filesystem::path&, const PitchDetectorConfig&,
                  const float* interleaved, size_t numSamples);
bool writeWavFile(const std::filesystem::path&, const RecordingData&);

// Returns nullopt if the file cannot be read, isn't a 32-bit float WAV, or
// doesn't carry a (consistent) config in its LIST INFO chunk.
std::optional<RecordingData> readWavFile(const std::filesystem::path&);

}  // namespace recording
}  // namespace saint
