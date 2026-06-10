#include <signal.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

#include "PitchDetector/Recording/ReplayPitchDetector.h"
#include "TunerDisplay.h"

namespace {
std::atomic<bool> gRunning{true};

void signalHandler(int) {
    gRunning = false;
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
        std::cerr << "  --fast  replay the entire file without waiting (default: real-time pace)"
                  << std::endl;
        return 1;
    }

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    const auto pitchDetector = saint::ReplayPitchDetector::fromFile(file);
    if (!pitchDetector) {
        std::cerr << "Could not load recording: " << file << std::endl;
        return 1;
    }

    const auto& config = pitchDetector->config();
    std::cout << "Replaying " << file << " (" << config.sampleRate << " Hz, block size "
              << config.samplesPerBlockPerChannel << ", " << pitchDetector->numBlocks()
              << " blocks). Press Ctrl+C to exit." << std::endl;
    std::cout << std::endl;

    saint::TunerDisplay display;
    const auto blockDuration =
        std::chrono::microseconds(1000000LL * config.samplesPerBlockPerChannel / config.sampleRate);
    while (gRunning && pitchDetector->numBlocksLeft() > 0) {
        const auto blockStart = std::chrono::steady_clock::now();
        display.update(pitchDetector->process(nullptr));
        if (!fast) {
            std::this_thread::sleep_until(blockStart + blockDuration);
        }
    }

    std::cout << std::endl << std::endl << "Replay finished." << std::endl;
    return 0;
}
