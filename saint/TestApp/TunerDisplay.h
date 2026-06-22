#pragma once

#include <string>

namespace saint {

class TunerDisplay {
   public:
    enum class State { NoPitch, Estimate, Hold };

    TunerDisplay();
    ~TunerDisplay();

    // Update the display with new pitch info
    // frequencyHz: 0 if no pitch detected
    // state: the algorithm's tracking state, drives the pitch-cursor colour (see State).
    // onsetDetected: true on the block where a note attack was detected. The indicator is
    //   latched for a short while so a single-block onset stays visible (see TunerDisplay.cpp).
    // status: optional text appended after the meter (e.g. a recording indicator)
    void update(float frequencyHz, State state, bool onsetDetected = false,
                const std::string& status = "");

    // Clear the display
    void clear();

   private:
    struct NoteInfo {
        std::string name;
        int octave;
        float cents;  // -50 to +50
    };

    static NoteInfo frequencyToNote(float frequencyHz);
    // Renders the meter; the needle (▼) is wrapped in needleColor..reset when non-empty.
    static std::string renderMeter(float cents, int width, const std::string& needleColor);

    float _lastFrequency = 0.f;
    // Counts down the remaining display updates the onset indicator stays lit for.
    int _onsetHoldFrames = 0;
    // When stdout isn't a terminal (e.g. a debugger console or a pipe), `\r` overwriting and
    // ANSI escapes don't work; fall back to plain, throttled line-per-update output.
    const bool _isTty;
    int _updateCount = 0;
};

}  // namespace saint
