#include "TunerDisplay.h"

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace saint {

namespace {
constexpr float kA4Frequency = 440.0f;
constexpr int kA4MidiNote = 69;

const char* kNoteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

// A note attack spans a single block (~11 ms at the TestApp's 512/44100 block rate), far too
// brief to see. Latch the onset indicator for this many display updates (~230 ms) so a pluck
// registers visually before it decays.
constexpr int kOnsetHoldFrames = 20;

// ANSI colour for the pitch cursor in each algorithm state.
const char* stateColorCode(TunerDisplay::State state) {
    switch (state) {
        case TunerDisplay::State::Estimate:
            return "\033[92m";  // green
        case TunerDisplay::State::Hold:
            return "\033[93m";  // yellow
        case TunerDisplay::State::NoPitch:
        default:
            return "\033[90m";  // grey
    }
}

const char* stateLabel(TunerDisplay::State state) {
    switch (state) {
        case TunerDisplay::State::Estimate:
            return "Estimated";
        case TunerDisplay::State::Hold:
            return "Held";
        case TunerDisplay::State::NoPitch:
        default:
            return "No pitch";
    }
}

// ANSI colour of the out-of-range banner: magenta, so it reads as neither a good reading
// (green) nor a stale one (yellow).
constexpr auto kOutOfRangeColor = "\033[95m";

// Appended to the state label so the line says which side of the range the pitch fell on.
const char* bucketSuffix(const std::optional<PitchBucket>& bucket) {
    if (bucket == PitchBucket::belowRange) {
        return " - below range";
    }
    if (bucket == PitchBucket::aboveRange) {
        return " - above range";
    }
    return "";
}
}  // namespace

TunerDisplay::TunerDisplay() : _isTty(isatty(fileno(stdout)) != 0) {
    if (_isTty) {
        // Hide cursor
        std::cout << "\033[?25l";
        std::cout << std::flush;
    }
}

TunerDisplay::~TunerDisplay() {
    if (_isTty) {
        // Show cursor
        std::cout << "\033[?25h";
        std::cout << std::flush;
    }
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

std::string TunerDisplay::renderMeter(float cents, int width, const std::string& needleColor) {
    std::ostringstream oss;

    // Clamp cents to [-50, 50]
    cents = std::max(-50.0f, std::min(50.0f, cents));

    // Calculate needle position (0 to width-1)
    const int centerPos = width / 2;
    const int needlePos = centerPos + static_cast<int>((cents / 50.0f) * centerPos);

    // Build the meter string
    for (int i = 0; i < width; ++i) {
        if (i == needlePos) {
            // The needle colour encodes the algorithm's state (see State); no colour when
            // stdout isn't a terminal.
            if (needleColor.empty()) {
                oss << "▼";
            } else {
                oss << needleColor << "▼\033[0m";
            }
        } else if (i == centerPos) {
            oss << "|";  // Center marker
        } else if (i < centerPos) {
            oss << (i == 0 ? "♭" : "-");
        } else {
            oss << (i == width - 1 ? "♯" : "-");
        }
    }

    return oss.str();
}

std::string TunerDisplay::renderOutOfRange(PitchBucket bucket, int width, bool useColor) {
    const auto tooLow = bucket == PitchBucket::belowRange;
    // ASCII, so its length in bytes is its width in columns.
    const std::string text = tooLow ? " TOO LOW - tune up " : " TOO HIGH - tune down ";
    const auto* arrow = tooLow ? "↑" : "↓";
    const auto textWidth = static_cast<int>(text.size());
    const auto left = std::max(0, (width - textWidth) / 2);
    const auto right = std::max(0, width - textWidth - left);

    std::ostringstream oss;
    if (useColor) {
        oss << kOutOfRangeColor;
    }
    // Emitted glyph by glyph rather than as a padded string: the arrows are multi-byte, so the
    // count that matters is columns, not bytes.
    for (auto i = 0; i < left; ++i) {
        oss << arrow;
    }
    oss << text;
    for (auto i = 0; i < right; ++i) {
        oss << arrow;
    }
    if (useColor) {
        oss << "\033[0m";
    }
    return oss.str();
}

void TunerDisplay::update(const PitchDetectionResult& result, State state, bool onsetDetected,
                          const std::string& status) {
    const auto frequencyHz = result.pitch;
    if (onsetDetected) {
        _onsetHoldFrames = kOnsetHoldFrames;
    }

    if (!_isTty) {
        // No in-place updating possible: print a plain line a couple of times per second.
        constexpr int kThrottle = 32;
        if (_updateCount++ % kThrottle != 0) {
            return;
        }
    } else {
        // Move cursor to beginning of line and clear it
        std::cout << "\r\033[K";
    }

    // Note-attack indicator, latched (see kOnsetHoldFrames) so a momentary onset is visible and
    // decays a few frames later. The line is cleared on every update, so an empty string when
    // idle leaves no residue.
    std::string onsetStr;
    if (_onsetHoldFrames > 0) {
        onsetStr = _isTty ? "  \033[93m● ONSET\033[0m" : "  ● ONSET";
        --_onsetHoldFrames;
    }

    // The pitch cursor (▼) is coloured by the algorithm state; no colour off a terminal.
    const std::string needleColor = _isTty ? stateColorCode(state) : "";

    // ----- line 1: the tuner meter -----
    if (frequencyHz <= 0.f) {
        // Same column layout as the pitch branch below, so that the meter doesn't shift when
        // detection toggles.
        std::cout << std::setw(3) << "--";
        std::cout << " │ ";
        std::cout << std::setw(6) << "---.-" << " Hz";
        std::cout << " │ ";
        std::cout << renderMeter(0, 41, needleColor);
        std::cout << " " << std::setw(3) << "--" << "¢";
    } else {
        const auto note = frequencyToNote(frequencyHz);
        // A pitch without a bucket is not supposed to happen (see PitchDetectionResult), and a
        // detector that leaves it unset means the in-range meter, not a crash.
        const auto bucket = result.bucket.value_or(PitchBucket::inRange);

        // Note name with octave (e.g., "A4", "C#3"), in the default text colour. Kept for an
        // out-of-range pitch too: naming it is honest, it is the pitch that is being played.
        std::ostringstream noteStr;
        noteStr << std::setw(2) << note.name << note.octave;

        std::cout << std::setw(3) << noteStr.str();
        std::cout << " │ ";
        std::cout << std::fixed << std::setprecision(1) << std::setw(6) << frequencyHz << " Hz";
        std::cout << " │ ";
        if (bucket == PitchBucket::inRange) {
            // Cents with sign, against the note the meter's needle points at.
            std::ostringstream centsStr;
            centsStr << std::showpos << std::fixed << std::setprecision(0) << std::setw(3)
                     << note.cents;
            std::cout << renderMeter(note.cents, 41, needleColor);
            std::cout << " " << centsStr.str() << "¢";
        } else {
            // No cents reading: it would be a deviation from a note that is not a tuning target
            // here, which is the one thing not to invite while the string is still out of range.
            std::cout << renderOutOfRange(bucket, 41, _isTty);
            std::cout << " " << std::setw(3) << "--" << "¢";
        }
    }
    std::cout << onsetStr;
    std::cout << status;

    // ----- line 2: the cursor + current-state label, redrawn in place -----
    const std::string cursor = needleColor.empty() ? "▼" : (needleColor + "▼\033[0m");
    if (_isTty) {
        // Drop to the next line, clear it, draw the indicator, then move back up so the next
        // update overwrites line 1 again. The cursor is hidden, so this doesn't flicker.
        std::cout << "\n\033[K" << cursor << " " << stateLabel(state) << bucketSuffix(result.bucket)
                  << "\033[A\r";
    } else {
        std::cout << "\n"
                  << "▼ " << stateLabel(state) << bucketSuffix(result.bucket) << "\n";
    }
    std::cout << std::flush;

    _lastFrequency = frequencyHz;
}

void TunerDisplay::clear() {
    if (_isTty) {
        // Clear line 1 and the state line below it, leaving the cursor back on line 1.
        std::cout << "\r\033[K\n\033[K\033[A\r" << std::flush;
    }
}

}  // namespace saint
