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

// Block size of the standard config that `readWavFile` synthesizes for WAV files that aren't app
// recordings. Matches the live app's block size (`runLive` in TestApp/main.cpp).
constexpr int defaultSamplesPerBlockPerChannel = 512;

std::string serializeConfig(const PitchDetectorConfig&);
std::optional<PitchDetectorConfig> deserializeConfig(const std::string&);

// Writes a 32-bit float WAV file with the config in the LIST INFO chunk. Clients receiving a
// `RecordingData` (see `IssueReportingPitchDetector::startIssueRecording`) must persist it with
// this function - and not a generic WAV writer - so that the metadata needed by
// `ReplayPitchDetector` is included.
bool writeWavFile(const std::filesystem::path&, const PitchDetectorConfig&,
                  const float* interleaved, size_t numSamples);
bool writeWavFile(const std::filesystem::path&, const RecordingData&);

// Reads a WAV file for replay.
//
// A native app recording - a 32-bit float WAV carrying a config in its LIST INFO chunk that is
// consistent with the format header - is returned verbatim, and `*warning` (if non-null) is left
// untouched.
//
// Any other readable WAV (e.g. 16-bit PCM, or a float file without our config) is assumed *not*
// to be an app recording: its samples are converted to float, a standard config is synthesized
// from the file's sample rate and channel count (`defaultSamplesPerBlockPerChannel`,
// `Tuning::Standard`), and `*warning` (if non-null) is set to a human-readable explanation.
//
// Returns nullopt only when the file cannot be read, is not a WAV, has more than two channels, or
// uses a sample format we cannot decode (see `decodeSamples`).
std::optional<RecordingData> readWavFile(const std::filesystem::path&,
                                         std::string* warning = nullptr);

}  // namespace recording
}  // namespace saint
