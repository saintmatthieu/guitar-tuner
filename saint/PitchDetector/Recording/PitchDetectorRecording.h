#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "PitchDetectorTypes.h"

namespace saint {
namespace recording {

struct PitchDetectorConfig {
    int sampleRate = 0;
    ChannelFormat channelFormat = ChannelFormat::Mono;
    int samplesPerBlockPerChannel = 0;
    Tuning tuning = Tuning::Standard;
    bool holdPitch = true;  // Hold the last pitch for a short time after the note stopped being
                            // detected. Avoids blinks in the UI.
};

struct RecordingData {
    PitchDetectorConfig config;
    std::vector<float> interleaved;
};

// The config is serialized as `key=value` pairs separated by `;` into the ICMT
// (comment) sub-chunk of the WAV file's LIST INFO chunk, so the file stays
// playable in any audio tool while carrying everything needed for replay.
// (Reading it back - `deserializeConfig`, `readWavFile` - is debug tooling and
// lives in PitchDetector/Test/RecordingFileReader.h, not in the production library.)
constexpr auto sampleRateKey = "sampleRate";
constexpr auto channelFormatKey = "channelFormat";
constexpr auto samplesPerBlockPerChannelKey = "samplesPerBlockPerChannel";
constexpr auto tuningKey = "tuning";
constexpr auto holdPitchKey = "holdPitch";

std::string serializeConfig(const PitchDetectorConfig&);

// Writes a 32-bit float WAV file with the config in the LIST INFO chunk. Clients receiving a
// `RecordingData` (see `IssueReportingPitchDetector::startIssueRecording`) must persist it with
// this function - and not a generic WAV writer - so that the metadata needed to replay it
// (see `ReplayPitchDetector`) is included.
bool writeWavFile(const std::filesystem::path&, const PitchDetectorConfig&,
                  const float* interleaved, size_t numSamples);
bool writeWavFile(const std::filesystem::path&, const RecordingData&);

}  // namespace recording
}  // namespace saint
