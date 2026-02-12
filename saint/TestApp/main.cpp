#include <signal.h>

#include <atomic>
#include <iomanip>
#include <iostream>

#include "AlsaAudioInput.h"
#include "PitchDetector/PitchDetectorFactory.h"
#include "TunerDisplay.h"

namespace {
std::atomic<bool> gRunning{true};

void signalHandler(int) {
    gRunning = false;
}
}  // namespace

int main(int argc, char* argv[]) {
    // Parse optional device argument
    std::string device = "default";
    if (argc > 1) {
        device = argv[1];
    }

    constexpr int kSampleRate = 44100;
    constexpr int kBlockSize = 512;

    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║                          Guitar Tuner                                    ║"
              << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣"
              << std::endl;
    std::cout << "║  Device: " << std::left << std::setw(63) << device << " ║" << std::endl;
    std::cout << "║  Sample rate: " << kSampleRate << " Hz, Block size: " << kBlockSize
              << " samples                          ║" << std::endl;
    std::cout << "║  Press Ctrl+C to exit                                                    ║"
              << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝"
              << std::endl;
    std::cout << std::endl;

    // Set up signal handler for graceful exit
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Create pitch detector
    auto pitchDetector = saint::PitchDetectorFactory::createInstance(
        kSampleRate, saint::ChannelFormat::Mono, kBlockSize);

    // Create display
    saint::TunerDisplay display;

    // Create audio input
    saint::AlsaAudioInput audioInput(kSampleRate, kBlockSize, device);

    // Start audio capture with callback
    bool success = audioInput.start([&](const float* samples, int numSamples) {
        if (!gRunning) {
            return;
        }

        const float frequency = pitchDetector->process(samples);
        display.update(frequency);
    });

    if (!success) {
        std::cerr << std::endl << "Failed to start audio capture." << std::endl;
        std::cerr << "Make sure you have ALSA configured and a microphone connected." << std::endl;
        std::cerr << std::endl;
        std::cerr << "Available devices can be listed with: arecord -L" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [device_name]" << std::endl;
        return 1;
    }

    // The capture loop runs until Ctrl+C
    audioInput.stop();

    std::cout << std::endl << std::endl << "Goodbye!" << std::endl;
    return 0;
}
