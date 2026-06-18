#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "Recording/PitchDetectorRecording.h"

// Reading recordings back is debug/test tooling, not part of the production PitchDetector
// library: it is used only by the offline ReplayApp and the recording tests. It lives here
// (and not under PitchDetector/Recording/) so the production library stays write-only - it
// hands clients a `RecordingData` and serializes it via `writeWavFile`; reading it again is
// somebody else's job.
namespace saint {
namespace recording {

// Block size of the standard config that `readWavFile` synthesizes for WAV files that aren't app
// recordings. Matches the live app's block size (`runLive` in TestApp/main.cpp).
constexpr int defaultSamplesPerBlockPerChannel = 512;

// Inverse of `serializeConfig`: parses the `key=value;...` comment string. Returns nullopt if a
// key is missing or a value doesn't parse.
std::optional<PitchDetectorConfig> deserializeConfig(const std::string&);

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
// uses a sample format we cannot decode.
std::optional<RecordingData> readWavFile(const std::filesystem::path&,
                                         std::string* warning = nullptr);

}  // namespace recording
}  // namespace saint
