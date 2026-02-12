#include "TunerDisplay.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace saint {

namespace {
constexpr float kA4Frequency = 440.0f;
constexpr int kA4MidiNote = 69;

const char* kNoteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
}  // namespace

TunerDisplay::TunerDisplay() {
    // Hide cursor
    std::cout << "\033[?25l";
    std::cout << std::flush;
}

TunerDisplay::~TunerDisplay() {
    // Show cursor
    std::cout << "\033[?25h";
    std::cout << std::flush;
}

TunerDisplay::NoteInfo TunerDisplay::frequencyToNote(float frequencyHz) {
    // Calculate MIDI note number (can be fractional)
    const float midiNote = 12.0f * std::log2(frequencyHz / kA4Frequency) + kA4MidiNote;

    // Round to nearest integer note
    const int nearestNote = static_cast<int>(std::round(midiNote));

    // Calculate cents deviation from nearest note
    const float cents = (midiNote - nearestNote) * 100.0f;

    // Extract note name and octave
    const int noteIndex = ((nearestNote % 12) + 12) % 12;
    const int octave = (nearestNote / 12) - 1;

    return {kNoteNames[noteIndex], octave, cents};
}

std::string TunerDisplay::renderMeter(float cents, int width) {
    std::ostringstream oss;

    // Clamp cents to [-50, 50]
    cents = std::max(-50.0f, std::min(50.0f, cents));

    // Calculate needle position (0 to width-1)
    const int centerPos = width / 2;
    const int needlePos = centerPos + static_cast<int>((cents / 50.0f) * centerPos);

    // Build the meter string
    for (int i = 0; i < width; ++i) {
        if (i == centerPos) {
            oss << "|";  // Center marker
        } else if (i == needlePos) {
            // Color the needle based on how close to center
            const float absCents = std::abs(cents);
            if (absCents < 5.0f) {
                oss << "\033[32m▼\033[0m";  // Green - in tune
            } else if (absCents < 15.0f) {
                oss << "\033[33m▼\033[0m";  // Yellow - close
            } else {
                oss << "\033[31m▼\033[0m";  // Red - off
            }
        } else if (i < centerPos) {
            oss << (i == 0 ? "♭" : "-");
        } else {
            oss << (i == width - 1 ? "♯" : "-");
        }
    }

    return oss.str();
}

void TunerDisplay::update(float frequencyHz) {
    // Move cursor to beginning of line and clear it
    std::cout << "\r\033[K";

    if (frequencyHz <= 0.f) {
        std::cout << "  --  │  ---.-  Hz  │  ";
        std::cout << renderMeter(0, 41);
        std::cout << std::flush;
        _lastFrequency = 0.f;
        return;
    }

    const auto note = frequencyToNote(frequencyHz);

    // Format note name with octave (e.g., "A4", "C#3")
    std::ostringstream noteStr;
    noteStr << std::setw(2) << note.name << note.octave;

    // Color the note name based on tuning accuracy
    const float absCents = std::abs(note.cents);
    std::string colorCode;
    if (absCents < 5.0f) {
        colorCode = "\033[32m";  // Green
    } else if (absCents < 15.0f) {
        colorCode = "\033[33m";  // Yellow
    } else {
        colorCode = "\033[31m";  // Red
    }

    // Format cents with sign
    std::ostringstream centsStr;
    centsStr << std::showpos << std::fixed << std::setprecision(0) << std::setw(3) << note.cents;

    // Print the tuner display
    std::cout << colorCode << std::setw(3) << noteStr.str() << "\033[0m";
    std::cout << " │ ";
    std::cout << std::fixed << std::setprecision(1) << std::setw(6) << frequencyHz << " Hz";
    std::cout << " │ ";
    std::cout << renderMeter(note.cents, 41);
    std::cout << " " << centsStr.str() << "¢";
    std::cout << std::flush;

    _lastFrequency = frequencyHz;
}

void TunerDisplay::clear() {
    std::cout << "\r\033[K" << std::flush;
}

}  // namespace saint
