# Low-note detection ("tuning up from slack") — idea catalogue

A design catalogue for making the tuner robust to the *post-string-change* use case: you
fit new strings and tune **up, up, up** from a very slack (far-below-target) string. While
the string is below the selected tuning's range, the detector — sized and gated for that
tuning — returns nothing, so the user gets no feedback while the note that matters is rising
toward the target.

Goal: report something useful for a string that is **below the lowest in-range note**. A
full frequency readout would be ideal (a rising number as you tune up); even a binary
**"too low"** indication is acceptable and may enable much lighter solutions.

Constraint: this is a real-time tuner, so anything added is on the audio thread and CPU
matters. A naïve "second full detector" roughly triples per-block FFT work.

---

## How the pipeline actually works (grounded in the code)

Establishing the cost structure and the control flow, because they decide which ideas are
viable.

**Cost driver — window length ∝ 1/minFreq.**
`FrequencyDomainTransformer::getWindowSizeSamples()` sizes the analysis window as
`numPeriods / minFreq` (`numPeriods ≈ 6` for the `MinimumThreeTerm` window), and the FFT is
the next power of two above it. So the per-block forward+inverse FFTs scale as **1/minFreq**:
halving minFreq doubles the window and ~doubles (often quadruples, across a pow-2 boundary)
the FFT cost. For Standard tuning `getMinFreq(Standard) = Db2 ≈ 69 Hz` → window ≈ 73 ms →
`fftSize = 4096`.

**The low lags are already computed.** `AutocorrPitchDetector::getXCorr` inverse-FFTs the
whole ACF into `_xcorr` (length `fftSize`), i.e. lags **0 … fftSize/2 ≈ 2048** (≈ 21 Hz) are
all present. But the peak search stops at `_lastSearchIndex = sampleRate/minFreq ≈ 538` (lag
of 69 Hz). **Lags ~538 … ~1275 (69 → 35 Hz) are computed and discarded.** Line 157's
`maximum /= _windowXcorr[maxIndex]` already overlap-corrects long lags, so a peak at lag 1275
(35 Hz) sits in a window with ~67% overlap remaining — still reliable. One octave below the
in-range floor is the practical reach before the window runs out.

**⚠️ The two ways `process()` returns 0 (the key correction).** There are *two* distinct
"no pitch" paths, and they are very different in frequency:

1. **Empty ACF search** — `AutocorrPitchDetector` finds no peak in range (`maxIndex == 0`).
   **Rare.** A slack string usually still produces *some* in-range ACF peak (a harmonic of
   the low note, or noise), so this path is seldom taken for the use case.
2. **The harmonicity / `probNotOctaviated` gate** — the ACF *did* return a non-zero
   `xcorrEstimate`, but `PitchDetectorImpl` later rejects it because
   `probNotOctaviated(presenceScore) < threshold` or `harmonicity < floor`. **This is the
   common way a low/out-of-range note is turned into "no pitch."**

   Any low-band fallback must therefore be triggered by **path 2 as well as path 1** — i.e.
   "the *whole in-range pipeline* produced no pitch," evaluated at the end of
   `PitchDetectorImpl::process`, not by an empty ACF search inside `AutocorrPitchDetector`.
   *(The first implementation attempt gated the fallback on path 1 only; it was effectively
   dead code for the use case and showed no consistent improvement in the test app.)*

**The disambiguator is hard-floored at the in-range minFreq.**
`AutocorrEstimateDisambiguator::disambiguateFundamentalIndex` skips every candidate below
`minF0 = _minFreq/_binFreq` and its `*2` candidate survives — so feeding it a genuine 50 Hz
estimate **octave-doubles it back into range**. A low-band estimate must bypass (or be given
a lowered `minF0` for) this stage.

**The benchmark gates are near-exact.** `checkReference` uses `|actual − reference| ≤ 0.01`
(absolute), not the ±1% the CLAUDE.md implies. So even a 0.3 c move in p99 RMS trips the
gate. (And the benchmark corpus has **no sub-E2 notes**, so it can only *guard against
regression* — it cannot measure a low-note feature's benefit.)

---

## Why `Tuning::Unknown` blows up — and the design rule it implies

The `Tuning::Unknown` experiment raised the 99th-percentile error to ~1900 cents.
**1900 c ≈ a perfect twelfth (3:1 = 1902 c) — the third harmonic.** So the wide config isn't
"bad at low notes"; it's that **widening the search range and the disambiguator prior makes
the detector lock onto the 3rd harmonic / a subharmonic on *normal, in-range* notes.** The
accuracy of the in-house detector *is* its narrow prior.

**Design rule for any hybrid:** never let the low-band path touch the in-range search range
or prior. Keep the in-range estimate byte-for-byte as today; consult the low band only as a
*separate, gated fallback* when the in-range pipeline yields nothing.

---

## Idea 1 — Extend the in-range search downward (near-zero CPU)

Don't add a detector; add loop iterations over the ACF that is **already computed**. When
the in-range pipeline yields no pitch, search the ACF at longer lags (down to ~one octave
below the in-range floor) with a **separate low-band acceptance test**, and return that
frequency (or a "too-low" flag). No new FFT.

- **Trigger (corrected):** fall back when `PitchDetectorImpl::process` would return 0 for
  *any* reason — most importantly when the `probNotOctaviated`/`harmonicity` gate rejects the
  in-range estimate — not only when the ACF search is empty.
- **Bypass the in-range disambiguator** for the low-band estimate (its `minF0` floor would
  octave-double it back into range).
- **Reach:** the Standard window (~73 ms) resolves down to ~35–40 Hz with the existing
  `_windowXcorr` overlap correction. That likely covers the whole realistic guitar
  tuning-up range (you rarely start below ~E1 ≈ 41 Hz). Below that needs Idea 2.

**Known failure mode — the sub-harmonic mirage.** On the *lowest in-range note* (E2 ≈ 82 Hz)
under heavy noise, the in-range path intermittently drops the note, and a low-band search
will gladly lock its **~41 Hz period-doubled ACF peak** → an octave-flat (1200 c) reading.
This is the same sub-fundamental contamination that makes a held note sag in its decay tail.
Presence alone won't separate it (a strong note's doubled-period peak is tall).

**The principled guard:** a genuine 41 Hz fundamental needs *odd* harmonics (41, 123, 205 …);
a real 82 Hz note read as 41 Hz has energy only on the *even* comb (82, 164 …). So a
harmonicity/octave check at the low-band hypothesis — the same logic the disambiguator
already uses, just floored at the low-band min instead of the in-range min — rejects the
mirage. **Reject the low-band estimate when its octave-up is itself a well-supported in-range
fundamental.** This is the missing piece that makes Idea 1 actually useful rather than a
source of octave errors.

---

## Idea 2 — Decimated low-band path (for coverage below the window's reach)

For notes below what the in-range window can resolve (drop tunings, bass, very slack
strings), don't clone the full-rate detector — **decimate**. Cost ∝ sampleRate × window
length; a low octave needs ~2× the window *time*, but at ¼ the rate that's ½ the samples, so
the low FFT is *smaller* than the main one. This turns "more than 2× CPU" into "main + a
fraction," and composes with the run-only-when-in-range-is-silent gate. The
already-low-passed signal makes anti-aliasing cheap. This is the natural production form of
the "second instance" plan, far cheaper than instantiating `Tuning::Unknown`.

---

## Idea 3 — One big-window FFT, dual search + dual prior (note, not recommendation)

Since one forward+inverse FFT yields the full-range ACF, a single transformer sized to the
low minFreq could feed two peak searches (narrow in-range prior; low-band prior) — one large
FFT instead of two detectors. Cheaper than two instances, **but** the in-range path then
inherits the long window's latency (`delaySamples = window/2` ~doubles) and the fitted
octaviation distributions would need refitting. Given the priority on FNR/responsiveness,
paying a latency tax on the in-range path is the wrong trade. Mentioned for completeness;
Idea 1 captures most of the upside without it.

---

## The "too-low indication is enough" angle

A binary flag unlocks lighter solutions, and Idea 1 delivers it almost for free. Two
cautions:

- **Base it on periodicity (ACF), never sub-band energy.** A slack low-E sits right on top
  of 50/60 Hz mains hum; an energy/centroid gate would false-trigger constantly, whereas an
  ACF/harmonic test will not.
- **You already have the Hz, so don't throw it away.** A valid low-band ACF peak gives the
  frequency; a continuously-rising readout ("55 Hz — keep going up") is far better UX than a
  binary "too low," at no extra cost. Aim for the frequency, degrade to the boolean only when
  confidence is low.

---

## Notes on the original two-instance plan

- **`bufferAudio()` / `transform()` split — good, do it.** `FrequencyDomainTransformer::process`
  currently couples buffering (insert/erase) with the FFT; splitting them lets the transform
  be called sporadically while buffering stays current. Keep the erase tied to buffering so
  the ring doesn't grow when the transform is skipped.
- **OnsetDetector ← minFreq coupling is questionable.** Onset is a broadband transient
  detector; its window should be tuned for time resolution, not minFreq. Decoupling +
  recalibrating is sound — but it's a production-phase yak-shave, not a PoC blocker.
- **Sharing onset + preprocessor — production, not PoC.** For a PoC, two full instances is
  simplest and a powerful machine absorbs it.
- **For the low instance, use a *bounded low band*, not `Tuning::Unknown`.** Unknown is
  exactly the config that produced the 1900 c blow-up. A bounded band (e.g. ~30–95 Hz, a few
  Hz of overlap above the in-range floor for crossover hysteresis) keeps the offending high
  partials outside its search.

---

## How to evaluate (what the benchmark can and can't tell us)

1. The benchmark is a **regression guard only** — no sub-E2 notes, so it cannot show a
   low-note feature's benefit. Validating the feature needs **low-note material**: either the
   **test app** (tune a slack string up from below E2 and watch the readout) or **adding
   sub-E2 recordings to the corpus** so FNR benefit and any low-band FPR are measured
   directly.
2. On the current corpus, a low-band fallback's *only* visible effect is downside (the E2
   sub-harmonic misfire nudging p99 RMS) until the octave guard above is in place — and even
   then it buys no measurable FNR because there are no low notes to rescue.
3. Suggested order: get the **trigger** right (fall back on gate rejection, not empty ACF) →
   add the **octave guard** → validate the actual benefit on **low-note material** → only then
   reach for Idea 2 if coverage below the window's reach is needed.
