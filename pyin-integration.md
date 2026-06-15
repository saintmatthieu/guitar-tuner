# pYIN integration

How the pYIN (probabilistic YIN) fundamental-frequency estimator is wired into the SAINT
benchmark. **Benchmark/TestApp only — never part of the production library** (pYIN is
GPL-2.0-or-later; the project is MIT). The upstream sources are fetched at build time, not
committed.

## Is it correct?

Validated three ways against independent references:

- **Glue** — the actual `PyinPitchDetector` (windowing, low-amp gate, fixed-lag trellis trimming,
  lag-delayed emission) matches a from-scratch reference pipeline bit-for-bit over a whole file
  (0 Hz max diff, 661/661 blocks of `e4.wav`).
- **Algorithm** — the pitch chain (YIN → MIDI → HMM observation → Viterbi → frequency), decoded
  *offline*, reproduces librosa's independent pYIN to <0.5 Hz (e4 329.5 vs 329.4, g3 97.8 vs 97.9);
  the FFT shim is confirmed by the `fast`/`slow` difference paths agreeing exactly.
- **Causal track** — against the actual Vamp plugin (Sonic Visualiser, default parameters,
  hop 256 / lag 100) on `e4.wav`, the wrapper agrees frame-for-frame: same subharmonic plateau,
  jump to the fundamental at the same point (2.96 s vs 2.98 s), 99 % of frames within 5 Hz
  (median |diff| 0.08 Hz).

The only divergence is the final ~0.6 s, where the offline plugin flushes its lag-buffered frames
at end-of-stream — a per-block streaming detector structurally can't, and the note has decayed by
then anyway. Matching the plugin needs its default `lowampsuppression=0.1`; with low-amp
suppression off the causal jump time drifts, since the subharmonic-vs-fundamental likelihoods are
near-equal.

## What pYIN is

pYIN (Mauch & Dixon, 2014) runs in two stages: a *probabilistic YIN* front end that emits, per
analysis frame, a set of `(frequency, probability)` pitch candidates; and an **HMM** that
smooths those candidates into a pitch track. Reference: <https://github.com/c4dm/pyin>.

## What is vendored, and what is not

`saint/PitchDetector/Pyin/CMakeLists.txt` `FetchContent`s `c4dm/pyin` at tag **v1.2** and
compiles **only the algorithm core**: `Yin.cpp`, `YinUtil.cpp`, `MonoPitchHMM.cpp`,
`SparseHMM.cpp` (+ the header-only `MeanFilter.h`). Everything else (the Vamp plugin wrappers,
note segmentation `MonoNote*`) is skipped. Two upstream dependencies are stubbed out so no extra
libraries are pulled in (see `Pyin/shims/`):

- **`vamp-sdk/FFT.h`** — `YinUtil` uses `Vamp::FFTReal` for the FFT-accelerated autocorrelation.
  Rather than vendor the Vamp SDK, `shims/vamp-sdk/FFT.{h,cpp}` provides a small self-contained
  **double-precision** `Vamp::FFTReal` (radix-2). It only mimics the interface; it shares no Vamp
  code. (Verified to match the reference: forcing `fast=false`/`slowDifference` gives identical
  results.)
- **`boost/math/distributions.hpp`** — `MonoPitchHMM` `#include`s it but never uses a Boost
  symbol (the YIN threshold-distribution priors are precomputed static tables in `YinUtil.cpp`).
  `shims/boost/math/distributions.hpp` is an empty stub.

`pyin_core` (the upstream core + shims) is built as a separate static lib with the project's
strict warnings off and includes marked `SYSTEM`; the `PyinPitchDetector` wrapper is built with
the normal warning set.

## The streaming bridge (causal / fixed-lag)

The reference decodes pYIN's HMM **offline** (Viterbi over the whole file). The SAINT harness
drives detectors **per block** (`process()` returns one estimate per 10 ms block), so
`PyinPitchDetector` (`Pyin/PyinPitchDetector.cpp`) uses the reference `SparseHMM`'s built-in
**fixed-lag online decoding**. Per block:

1. Down-mix to mono and slide a rolling `frameSize`-sample window (hop = blockSize).
2. `Yin::processProbabilisticYin(window)` → candidate `freqProb`.
3. Convert each candidate Hz → MIDI (`12·log2(f/440)+69`) and apply pYIN's low-amplitude
   suppression — exactly as the reference `PYinVamp` does — then `MonoPitchHMM::calculateObsProb`.
4. `SparseHMM::initialise`/`process` advance the fixed-lag trellis; once it fills, emit the
   Viterbi-optimal pitch of the frame `fixedLag-1` blocks in the past
   (`track().front()` → `nearestFreq`; a negative result means unvoiced → return `0 Hz`).
5. `presenceScore` = the YIN voiced-probability mass of that frame (feeds the ROC/AUC).

`delaySamples() = frameSize/2 + (fixedLag-1)·blockSize`.

### CLI knobs (`createPyin` in `BenchmarkAlgorithms.cpp`)

| knob | default | meaning |
|---|---|---|
| `pyinFrameSize` | 2048 | analysis window (power of two) |
| `pyinFixedLag` | 20 | HMM smoothing lag in frames; latency = `(n-1)` blocks |
| `pyinThreshDistr` | 2 | YIN threshold-distribution prior (0 uniform; 1–4 Beta mean .10/.15/.20/.30) |
| `pyinLowAmp` | 0.1 | block-RMS below which candidates are suppressed |

pYIN does its own HMM smoothing, so — like the other third-party detectors — it is benchmarked
**raw**, without the in-house median filter + smoother.

## Important caveat: causal vs offline

This is a **causal** integration, and that is a deliberate choice (the tuner is real-time;
pYIN's offline Viterbi is not usable live). The trade-off is real and was verified:

- **Offline full Viterbi** of this same vendored core reproduces librosa's pYIN exactly
  (e.g. `e4.wav` → 329 Hz, correct).
- **Causal fixed-lag** *cannot* match it for sustained notes: per-frame, pYIN's front end
  spreads probability near-equally across a note's subharmonics (T, T/2, T/3 …), and only
  full-file Viterbi disambiguates. A bounded-latency decode locks onto a subharmonic
  (`e4.wav` → 109 Hz ≈ E4/3). No `fixedLag` value fixes this (swept to 200).

So `pyin`'s benchmark numbers reflect what a real-time tuner could achieve with pYIN, **not**
pYIN's published offline accuracy. See `pyin-benchmark-results.md`.

## Files

```
saint/PitchDetector/Pyin/
  CMakeLists.txt                      option + FetchContent + pyin_core + wrapper
  PyinPitchDetector.{h,cpp}           the saint::PitchDetector wrapper (causal fixed-lag)
  shims/vamp-sdk/FFT.{h,cpp}          self-contained double-precision Vamp::FFTReal
  shims/boost/math/distributions.hpp  empty stub (unused include)
```
Wired in: `saint/PitchDetector/CMakeLists.txt` (`add_subdirectory(Pyin)`),
`saint/PitchDetector/Test/CMakeLists.txt` and `BenchmarkAlgorithms.cpp`.
