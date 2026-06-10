#pragma once

#include <string>

namespace saint {

class TunerDisplay {
   public:
    TunerDisplay();
    ~TunerDisplay();

    // Update the display with new pitch info
    // frequencyHz: 0 if no pitch detected
    // status: optional text appended after the meter (e.g. a recording indicator)
    void update(float frequencyHz, const std::string& status = "");

    // Clear the display
    void clear();

   private:
    struct NoteInfo {
        std::string name;
        int octave;
        float cents;  // -50 to +50
    };

    static NoteInfo frequencyToNote(float frequencyHz);
    static std::string renderMeter(float cents, int width, bool useColor);

    float _lastFrequency = 0.f;
    // When stdout isn't a terminal (e.g. a debugger console or a pipe), `\r` overwriting and
    // ANSI escapes don't work; fall back to plain, throttled line-per-update output.
    const bool _isTty;
    int _updateCount = 0;
};

}  // namespace saint
