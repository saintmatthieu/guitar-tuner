#pragma once

#include <optional>
#include <string>

#include "PitchDetector/PitchDetector.h"

namespace saint {

class TunerDisplay {
   public:
    enum class State { NoPitch, Estimate, Hold };

    TunerDisplay();
    ~TunerDisplay();

    // Update the display with new pitch info
    // result: the detector's estimate. A `belowRange`/`aboveRange` bucket replaces the meter
    //   with a "too low"/"too high" banner - the needle means "tune to this note", and a pitch
    //   outside the tuning's range has no note to tune to; the frequency readout stays, since
    //   watching it climb is the point while tuning a slack string up.
    // state: the algorithm's tracking state, drives the pitch-cursor colour (see State).
    // onsetDetected: true on the block where a note attack was detected. The indicator is
    //   latched for a short while so a single-block onset stays visible (see TunerDisplay.cpp).
    // status: optional text appended after the meter (e.g. a recording indicator)
    void update(const PitchDetectionResult& result, State state, bool onsetDetected = false,
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
    // Renders, in the meter's place and to the same width, which way an out-of-range pitch has
    // to move to get into range.
    static std::string renderOutOfRange(PitchBucket bucket, int width, bool useColor);

    float _lastFrequency = 0.f;
    // Counts down the remaining display updates the onset indicator stays lit for.
    int _onsetHoldFrames = 0;
    // When stdout isn't a terminal (e.g. a debugger console or a pipe), `\r` overwriting and
    // ANSI escapes don't work; fall back to plain, throttled line-per-update output.
    const bool _isTty;
    int _updateCount = 0;
};

}  // namespace saint
