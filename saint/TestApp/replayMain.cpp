#include <signal.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

#include "ReplayPitchDetector.h"
#include "TunerDisplay.h"

#ifdef SAINT_REPLAY_WITH_ALSA
#include "AlsaAudioOutput.h"
#endif

namespace {
std::atomic<bool> gRunning{true};

void signalHandler(int) {
    gRunning = false;
}

bool wasOnset(const saint::DebugOutput& debug) {
    const auto it = debug.find("isOnset");
    return it != debug.end() && it->second != 0.f;
}
}  // namespace

int main(int argc, char* argv[]) {
    auto fast = false;
    std::filesystem::path file;
    auto validArgs = true;
    for (auto i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fast") {
            fast = true;
        } else if (file.empty()) {
            file = arg;
        } else {
            validArgs = false;
        }
    }
    if (file.empty() || !validArgs) {
        std::cerr << "Replays an issue recording saved by the guitar tuner (TestApp)."
                  << std::endl;
        std::cerr << "Usage: " << argv[0] << " [--fast] <recording.wav>" << std::endl;
        std::cerr << "  --fast  replay the entire file without waiting and without audio "
                     "(default: real-time pace"
#ifdef SAINT_REPLAY_WITH_ALSA
                     ", playing the recorded audio"
#endif
                     ")"
                  << std::endl;
        return 1;
    }

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    std::string warning;
    const auto pitchDetector = saint::ReplayPitchDetector::fromFile(file, &warning);
    if (!pitchDetector) {
        std::cerr << "Could not load recording: " << file << std::endl;
        return 1;
    }
    if (!warning.empty()) {
        std::cerr << "Warning: " << warning << std::endl;
    }

    const auto& config = pitchDetector->config();
    std::cout << "Replaying " << file << " (" << config.sampleRate << " Hz, block size "
              << config.samplesPerBlockPerChannel << ", " << pitchDetector->numBlocks()
              << " blocks). Press Ctrl+C to exit." << std::endl;
    std::cout << std::endl;

    saint::TunerDisplay display;
    const auto blockDuration =
        std::chrono::microseconds(1000000LL * config.samplesPerBlockPerChannel / config.sampleRate);

#ifdef SAINT_REPLAY_WITH_ALSA
    // In real-time mode, play the recorded audio through the speakers. The
    // blocking writes pace the loop, so no extra sleep is needed while playing.
    const auto framesPerBlock = config.samplesPerBlockPerChannel;
    std::optional<saint::AlsaAudioOutput> player;
    bool playing = false;
    if (!fast) {
        player.emplace(config.sampleRate, numChannels(config.channelFormat), framesPerBlock);
        if (player->open()) {
            playing = true;
        } else {
            std::cerr << "Continuing without audio playback." << std::endl;
            player.reset();
        }
    }
#endif

    while (gRunning && pitchDetector->numBlocksLeft() > 0) {
#ifdef SAINT_REPLAY_WITH_ALSA
        if (playing) {
            const float* block = pitchDetector->peekBlock();
            saint::DebugOutput debug;
            const float frequency = pitchDetector->process(nullptr, &debug);
            display.update(frequency, wasOnset(debug));
            if (!player->write(block, framesPerBlock)) {
                // Playback died mid-stream; fall back to timed pacing.
                playing = false;
                player.reset();
            }
            continue;
        }
#endif
        const auto blockStart = std::chrono::steady_clock::now();
        saint::DebugOutput debug;
        const float frequency = pitchDetector->process(nullptr, &debug);
        display.update(frequency, wasOnset(debug));
        if (!fast) {
            std::this_thread::sleep_until(blockStart + blockDuration);
        }
    }

    std::cout << std::endl << std::endl << "Replay finished." << std::endl;
    return 0;
}
