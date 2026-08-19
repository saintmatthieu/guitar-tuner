#pragma once

#include <vector>

#include "Autocorrelation.h"
#include "ButterworthFilter.h"
#include "DummyPitchDetectorLogger.h"
#include "FrequencyDomainTransformer.h"
#include "PitchDetectorTypes.h"

namespace saint {
class PitchDetectorLoggerInterface;

/**
 * @brief Long-window view of the octave below the search range, for the below-range ("too low")
 * verdict.
 *
 * The in-range analysis window is sized for the in-range floor - around 76 ms for a standard
 * tuning - and that is too short to say anything about what lies below it: the harmonics of a
 * 55 Hz string are spaced closer than the window's own main lobe, so its comb cannot be told
 * from the comb of the note an octave above, nor from the mains hum a slack low E sits on
 * (50 Hz and 54.7 Hz are 0.4 bins apart there). Measurements in
 * low-band-detection-results.md.
 *
 * The window length that question needs - two to three times the in-range one - is affordable
 * because the question is also band-limited: a below-range fundamental and the harmonics that
 * identify it all sit under a few hundred Hz, so the analysis runs on a further-decimated
 * signal, where a longer window in *time* is fewer samples than the in-range FFT. Its extra
 * latency falls only on the below-range verdict, which nobody tunes against note by note.
 */
class LowBandAnalyzer {
   public:
    /**
     * @param sampleRate rate of the blocks fed to @ref process, i.e. the preprocessor's
     * (already decimated) rate, not the input rate.
     * @param samplesPerBlockPerChannel nominal frames per @ref process call at that rate.
     * @param minFreq lower edge of the in-range search range: the analysis spans the octave
     * below it.
     * @param logger receives this analysis's own keys (`lowBand*`) for the one frame the logger
     * records; plot them with eval/showLowBandAnalysis.py.
     */
    LowBandAnalyzer(int sampleRate, ChannelFormat, int samplesPerBlockPerChannel, float minFreq,
                    PitchDetectorLoggerInterface& logger, LowBandConfig config = {});

    /**
     * @brief Consumes one block of the preprocessor's output and updates the analysed spectrum.
     * Must be called once per block, whether or not a verdict is wanted, to keep the internal
     * buffering in step with the input.
     */
    void process(const std::vector<float>& block);

    struct Verdict {
        /// Best-fitting fundamental in the octave below the range, 0 if the band is empty.
        float frequency = 0.f;
        /// The share of its harmonic comb that the in-range estimate cannot account for. This
        /// is the discriminating evidence: a string really sounding below the range fills in
        /// harmonics that the in-range estimate says nothing about, whereas if that estimate is
        /// itself the fundamental, every partial found down here is one of its own.
        float support = 0.f;
    };

    /**
     * @brief How periodic the band below the range is, at its most periodic period: the same
     * measure the in-range detector calls its presence score (@ref findAutocorrPeak), read over
     * the below-range lags. Unlike the in-range one it is computed on a window long enough for
     * those periods. Left by the last @ref process call.
     */
    float presence() const {
        return _presence;
    }

    /**
     * @brief Everything @ref below weighed, for the one frame the logger records. Assembling it
     * is a few dozen floats' worth of bookkeeping, so @ref below only does it when asked -
     * pass nullptr on the real-time path. The vectors keep their capacity between calls.
     */
    struct Diagnostics {
        // One entry per candidate considered.
        std::vector<float> candidateHz;
        std::vector<float> candidateSupport;
        std::vector<float> candidateProminence;  // total comb prominence, what the fit maximises
        // The winning candidate's comb: where each harmonic was looked for, what was found
        // there, and whether the in-range estimate already accounted for it (1) or not (0).
        std::vector<float> combHz;
        std::vector<float> combProminence;
        std::vector<float> combExplained;

        void clear();
    };

    /**
     * @brief Looks for a fundamental below the search range, and reports how much of it
     * `inRangeEstimate` fails to explain. Reads the spectrum left by the last @ref process
     * call.
     *
     * Searching for the fundamental rather than assuming it to be `inRangeEstimate / k`
     * matters: the in-range estimate need not be an integer harmonic of the string sounding
     * below. A1_1 in the corpus mostly reads 82 Hz - its strongest partial is the 3rd, at
     * 164 Hz, and the autocorrelation settles on half of that - so the fundamental is two
     * thirds of the estimate, which no integer k reaches.
     */
    Verdict below(float inRangeEstimate, Diagnostics* diagnostics = nullptr) const;

    /// Latency of the analysed spectrum, in samples at the rate passed to the constructor.
    int delaySamples() const {
        return _decimationFactor * _transformer.delaySamples();
    }

   private:
    static constexpr int filterOrder = 6;

    const int _decimationFactor;
    const ChannelFormat _channelFormat;
    const int _rate;
    // The inner transformer logs the same keys as the in-range one (rate, FFT size, spectrum),
    // with different values, so it gets a sink of its own rather than corrupting the in-range
    // path's diagnostics. This analysis logs under its own `lowBand*` keys instead.
    DummyPitchDetectorLogger _transformerLogger;
    PitchDetectorLoggerInterface& _logger;
    // Anti-alias filter for the further decimation. The preprocessor's own 5 kHz low-pass is
    // far above this Nyquist, so without it everything from here to 5 kHz would fold back into
    // the band being analysed.
    ButterworthFilter<filterOrder> _lowpass;
    FrequencyDomainTransformer _transformer;
    const float _binFreq;
    const float _minFrequency;
    const float _rangeFloor;
    // Highest harmonic order the in-range estimate is allowed to be of the fundamental looked
    // for: the denominator of the ratios searched (see below()). Keeping it low is what keeps
    // high notes out of this entirely - the higher the estimate, the more rational
    // sub-multiples of it land in the band, and the likelier one of them lines up with the
    // harmonics of the mains hum instead of with a string.
    const int _maxHarmonic;
    // Top of the band the noise floor is estimated over: as far up as the comb reaches.
    const int _floorBandEnd;

    // Autocorrelation of the band, for the presence score. The band is already limited by the
    // anti-alias filter, so the low-pass the autocorrelation takes is flat.
    RealFft _fft;
    const std::vector<float> _acfLpWindow;
    const std::vector<float> _windowXcorr;
    const int _firstLag;
    const int _lastLag;
    float _presence = 0.f;

    // Reused buffers, so process() allocates nothing on the audio thread.
    std::vector<float> _decimated;
    std::vector<float> _xcorr;
    std::vector<std::complex<float>> _freqScratch;
    std::vector<float> _spectrum;  // dB power spectrum, this frame's noise floor at 0
    std::vector<float> _floorScratch;
    int _decimationPhase = 0;
};
}  // namespace saint
