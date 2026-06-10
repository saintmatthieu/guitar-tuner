#include <poll.h>
#include <signal.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>

#include "AlsaAudioInput.h"
#include "PitchDetector/PitchDetectorFactory.h"
#include "PitchDetector/Recording/IRecordingListener.h"
#include "PitchDetector/Recording/IssueReportingPitchDetector.h"
#include "TunerDisplay.h"

namespace {
std::atomic<bool> gRunning{true};

void signalHandler(int) {
    gRunning = false;
}

// Puts stdin in non-canonical, no-echo mode so single key presses can be
// polled from the audio callback without blocking.
class RawStdin {
   public:
    RawStdin() {
        if (tcgetattr(STDIN_FILENO, &_original) != 0) {
            return;
        }
        termios raw = _original;
        raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO);
        // In non-canonical mode these slots control read(): without resetting them, VMIN
        // inherits the VEOF character (4), making read() block until 4 bytes arrive - which
        // freezes the audio loop after a key press and even eats Ctrl+C (the blocked read is
        // restarted after the signal handler ran).
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        _active = tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0;
    }

    ~RawStdin() {
        if (_active) {
            tcsetattr(STDIN_FILENO, TCSANOW, &_original);
        }
    }

    std::optional<char> getKey() const {
        pollfd fd{STDIN_FILENO, POLLIN, 0};
        if (::poll(&fd, 1, 0) > 0 && (fd.revents & POLLIN)) {
            char c;
            if (read(STDIN_FILENO, &c, 1) == 1) {
                return c;
            }
        }
        return std::nullopt;
    }

   private:
    termios _original{};
    bool _active = false;
};

std::filesystem::path makeRecordingPath() {
    const auto now = std::time(nullptr);
    std::ostringstream oss;
    oss << "tuner-recording-" << std::put_time(std::localtime(&now), "%Y%m%d-%H%M%S") << ".wav";
    return oss.str();
}

// Tracks the recording countdown (shown in the tuner line via status()) and writes the WAV file
// on completion. In this console app the file can be written right on the audio thread; a
// real-time client would rather hand the data over to a worker thread.
class ConsoleRecordingListener : public saint::IRecordingListener {
   public:
    explicit ConsoleRecordingListener(saint::TunerDisplay& display) : _display(display) {}

    void setPath(std::filesystem::path path) {
        _path = std::move(path);
    }

    void onProgress(int remainingSeconds) override {
        _remainingSeconds = remainingSeconds;
    }

    void onComplete(saint::recording::RecordingData data) override {
        _remainingSeconds.reset();
        _display.clear();
        if (saint::recording::writeWavFile(_path, data)) {
            std::cout << "Issue report saved to " << _path << " - replay it with: ReplayApp "
                      << _path.string() << std::endl;
        } else {
            std::cout << "Failed to save issue report to " << _path << std::endl;
        }
    }

    std::string status() const {
        if (!_remainingSeconds.has_value()) {
            return "";
        }
        std::ostringstream oss;
        oss << "  \033[31m● REC " << *_remainingSeconds << " s\033[0m";
        return oss.str();
    }

   private:
    saint::TunerDisplay& _display;
    std::filesystem::path _path;
    std::optional<int> _remainingSeconds;
};

int runLive(const std::string& device, const std::optional<std::filesystem::path>& outPath,
            const char* appName) {
    constexpr int kSampleRate = 44100;
    constexpr int kBlockSize = 512;
    constexpr int kIssueRecordingSeconds = 10;

    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║                          Guitar Tuner                                    ║"
              << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣"
              << std::endl;
    std::cout << "║  Device: " << std::left << std::setw(63) << device << " ║" << std::endl;
    std::cout << "║  Sample rate: " << kSampleRate << " Hz, Block size: " << kBlockSize
              << " samples                          ║" << std::endl;
    std::ostringstream reportLine;
    reportLine << "  Press 'r' to report an issue (records the next " << kIssueRecordingSeconds
               << " s for offline replay)";
    std::cout << "║" << std::left << std::setw(74) << reportLine.str() << "║" << std::endl;
    std::cout << "║  Press Ctrl+C to exit                                                    ║"
              << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝"
              << std::endl;
    std::cout << std::endl;

    auto pitchDetector = saint::PitchDetectorFactory::createInstance(
        kSampleRate, saint::ChannelFormat::Mono, kBlockSize);

    saint::TunerDisplay display;
    ConsoleRecordingListener recordingListener(display);
    RawStdin rawStdin;
    saint::AlsaAudioInput audioInput(kSampleRate, kBlockSize, device);

    bool success = audioInput.start([&](const float* samples, int numSamples) {
        if (!gRunning) {
            // start() blocks in the capture loop, which only watches its own
            // flag; stop() breaks it so start() returns and we exit on Ctrl+C.
            audioInput.stop();
            return;
        }

        if (const auto key = rawStdin.getKey(); key == 'r' || key == 'R') {
            recordingListener.setPath(outPath.value_or(makeRecordingPath()));
            pitchDetector->startIssueRecording(kIssueRecordingSeconds, recordingListener);
        }

        const float frequency = pitchDetector->process(samples);
        display.update(frequency, recordingListener.status());
    });

    if (!success) {
        std::cerr << std::endl << "Failed to start audio capture." << std::endl;
        std::cerr << "Make sure you have ALSA configured and a microphone connected." << std::endl;
        std::cerr << std::endl;
        std::cerr << "Available devices can be listed with: arecord -L" << std::endl;
        std::cerr << "Usage: " << appName << " [device_name] [--out <recording.wav>]"
                  << std::endl;
        return 1;
    }

    // The capture loop runs until Ctrl+C
    audioInput.stop();

    std::cout << std::endl << std::endl << "Goodbye!" << std::endl;
    return 0;
}
}  // namespace

int main(int argc, char* argv[]) {
    std::string device = "default";
    std::optional<std::filesystem::path> outPath;
    for (auto i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out") {
            if (i + 1 == argc) {
                std::cerr << "--out requires a file name" << std::endl;
                std::cerr << "Usage: " << argv[0] << " [device_name] [--out <recording.wav>]"
                          << std::endl;
                return 1;
            }
            outPath = argv[++i];
        } else {
            device = arg;
        }
    }

    // Set up signal handler for graceful exit
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    return runLive(device, outPath, argv[0]);
}
