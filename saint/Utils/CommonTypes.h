#pragma once

#include <optional>

namespace juce {
class MidiFile;
class MidiMessage;
class MidiMessageSequence;
}  // namespace juce

namespace saint {

class IntervalGetter;
class PositionGetter;

enum class PlayheadCommand {
    play,
    pause,
    stop,
};

struct PlayedNote {
    int noteNumber;
    // If not present then no harmonization.
    std::optional<int> interval = std::nullopt;
};

struct IntervalSpan {
    float beginCrotchet;
    std::optional<PlayedNote> playedNote;
};

bool operator==(const PlayedNote& a, const PlayedNote& b);
bool operator==(const IntervalSpan& a, const IntervalSpan& b);
bool operator!=(const PlayedNote& a, const PlayedNote& b);
bool operator!=(const IntervalSpan& a, const IntervalSpan& b);

struct MidiNoteMsg {
    float crotchet;
    bool isNoteOn;
    int noteNumber;
};

struct Fraction {
    int num;
    int den;
};
bool operator==(const Fraction&, const Fraction&);
bool operator!=(const Fraction&, const Fraction&);

struct Position {
    int barIndex;
    float beatIndex;
};
bool operator==(const Position&, const Position&);
bool operator!=(const Position&, const Position&);

struct TimeSignaturePosition {
    int barIndex;
    float crotchet;
    Fraction timeSignature;
};
}  // namespace saint
