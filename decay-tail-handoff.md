# Decay-tail pitch sag — investigation & WIP handoff

_Last updated 2026-06-23. Hand-off note for picking this work up in a fresh session._

## TL;DR — where things stand

- **Branch `pitch-detection-tail-improvements`, HEAD `5d48002`** ("Stricter harmonicity gate", = the locked-phase gate moved **0.3 → 0.5**). Goldens are re-seeded to the 0.5 operating point (`median_RMS 2.5268`, `p99_RMS 39.5605`, `FNR 0.2224`).
- **The experimental code below is live in the working tree, staged** (`git status` shows it staged; `git diff --cached` is the full diff). A `git reset --soft 5d48002` un-committed the WIP but kept its whole tree staged. It is **also** backed up in commit **`bc5e561` ("backup")** (currently off any branch tip — `git branch decay-tail-wip bc5e561` to keep it). Everything is **default-OFF**, so it does not change shipping behaviour; it only adds toggles/knobs.

## The problem

`tuner-recording-20260622-110920.wav` (E4, mono 44.1 k, block 512, in the repo root): the note is detected correctly, then ~2.2–3.2 s in the **displayed pitch sags ~10 cents flat** (−4 c → −13.7 c) and finally drops to no-pitch at 3.23 s.

## Root cause (confirmed)

The true pitch is **steady at ~−4 c** the whole time (LS sinusoid fit + zoomed spectrum agree; the player tuned slightly flat). The sag is an **algorithm artifact**:

- In the decay tail, **non-harmonic low-frequency energy grows** relative to the decaying fundamental — room rumble / mains-hum region ~55–185 Hz plus a ~277–279 Hz component (≈ a tone below E4); by ~2.2 s a ~58 Hz component dominates the band.
- The autocorrelation is computed over the whole **0–1500 Hz** band (`lpWindow`), so this **sub-fundamental energy pulls the ACF first-peak toward longer lag** → a flat (low) bias.
- Chain to the display: biased per-frame estimate → median filter latches it as `_heldPitch` → heavy output smoother (`PitchDetectionSmoother`, C=0.95, ~200 ms time-constant) glides the displayed value down to it → hold cap (`_maxHoldFrames`) reached → no-pitch.

**Ruled out** (don't re-investigate):
- _Lag quantization / parabolic interp._ One integer ACF lag ≈ 12.8 c at E4, but the parabola was already accurate. Implemented freq-domain zero-padding of |X|² before the IFFT (×4, the `autocorrUpsamplingFactor` constant): changed the estimate **< 0.05 c**, no benchmark gain, ~2× CPU. Reverted.
- The FFT-size commits (`472031e`, `fe5bb15`); the −60 dB sample gate in `FrequencyDomainTransformer`; cross-frame ACF averaging (it _helps_ — disabling makes the sag worse).

Proven with Python replicas of the exact ACF (MTT window 3372 samples, FFT 4096, `lpWindow` 1500 Hz): a band-pass around just the fundamental (285–380 Hz) restores the peak to −3…−6 c where the full band gives −19…−31 c.

## WIP #1 — constraint band-pass (in `bc5e561`, default OFF; ON in TestApp)

**What:** when locked, band-pass the power spectrum to a window around the constrained fundamental before the inverse FFT, to reject the sub-fundamental contamination. It is a **zero-phase spectral mask recomputed per block** — _not_ a recursive filter — so it cannot go unstable as the centre frequency tracks a bending pitch. Fundamental-only, **±2 semitones**, centred on the constraint, engaged **only when** sub-fundamental/in-band energy ratio exceeds a threshold (so healthy frames stay broadband). CPU is essentially free (just a mask on the existing FFT; 0.95 vs 0.92 in the realtime log).

**Code (all in `bc5e561`):**
- `AutocorrPitchDetector` — `applyConstraintBandPass` ctor flag (default false); `applyBandPass()` + `getXCorr()` band-pass; energy-ratio trigger in `process()`.
- `PitchDetectorTypes.h` — `autocorrConstraintBandHalfWidthSemitones = 2`, `autocorrConstraintBandContaminationRatio = 0.15`, `autocorrSubFundamentalFloorHz = 40`.
- `PitchDetectorFactory::createInstance(..., bool applyConstraintBandPass = false)` plumbs it; **`TestApp/main.cpp` passes `true`** so it can be auditioned live (`./build/Release/saint/TestApp/TestApp`).

**Result:** fixes the sag (gated: −13.7 → −8.7 c; healthy region bit-identical to baseline). **But a real tradeoff** (measured when HEAD was still 0.3 — _needs re-benchmarking on the 0.5 baseline_): gated@0.15 gave AVG 1.69→1.09 ✓, p99 59.5→54.2 ✓, **median 2.60→3.41 ✗**, FNR 0.206→0.214. Band-passing close to the contaminant clips the fundamental's lower spectral skirt → ~+1 c sharp bias on every engaged frame. Hence default-OFF; ON in TestApp only.

## WIP #2 — harmonic-lag consistency gate (in `bc5e561`, default OFF — NEGATIVE RESULT)

**Idea (user's):** a locked-phase release that distinguishes a _true pitch shift_ (peg turn — must keep tracking) from a _noise wander_ (must drop), which position-vs-constraint alone cannot. The ACF of a periodic signal peaks at every integer multiple of the fundamental lag L; the **2L peak sits at exactly 2·L₁**. A genuine shift moves the whole structure coherently (2L stays at 2·L₁, consistency ≈ 0 at any pitch); contamination pulls L₁ without moving 2L, so `|L2 − 2·L₁|` flags a wandered estimate.

**Mechanism validated** (Python + C++): sub-cent on healthy frames and on clean _shifted_ tones (peg-turns don't trip it); blows up to −7…−25 c through the sag.

**Code (all in `bc5e561`):**
- `AutocorrPitchDetector::process(..., float* harmonicConsistencyCents, float* secondaryPeakScore)` — finds the 2L peak (`refinedPeakIn`, fully bounds-guarded; clamp parabolic offset to ±1 or a flat peak segfaults), outputs the cents deviation + normalised 2L-peak height.
- `OctaviationGateConfig.lockedConsistencyCents` (0 = off → presence cut) and `.lockedSecondaryPeakFloor = 0.2`; gate logic in `PitchDetectorImpl::process()` releases when `|consistency| > lockedConsistencyCents` or `secondaryPeakScore < floor`.
- CLI knobs: `lockedConsistencyCents=`, `lockedSecondaryPeakFloor=`.

**Result (NEGATIVE — do not adopt as the gate):** at matched FNR the presence cut has the **lower p99 every time** (FNR≈0.222: p99 51 vs 40; FNR≈0.24: 37 vs 26); median is a wash (≤0.02 c better). **Why:** p99 is dominated by _coherent_ lock-drift / octave-class errors that keep their harmonic structure and so **pass** the consistency check; consistency only catches the gentler incoherent sag (~p75–p95 band), which neither gated metric (median, p99) is sensitive to. The thing that makes it elegant (invariant to coherent change) is why it can't catch the coherent errors that own the tail.

Consistency-gate sweep (secondaryPeakFloor 0.2) vs presence sweep:

| consistency | FNR | median | p99 |   | presence | FNR | median | p99 |
|---|---|---|---|---|---|---|---|---|
| 5 c | 0.239 | 2.433 | 37.3 |   | 0.7 | 0.243 | 2.442 | 26.0 |
| 7 c | 0.224 | 2.527 | 51.4 |   | 0.5 | 0.222 | 2.527 | 39.6 |
| 10 c | 0.210 | 2.584 | 63.3 |   | 0.3 | 0.206 | 2.608 | 59.5 |
| 20 c | 0.192 | 2.737 | 90.2 |   | 0.2 | 0.200 | 2.69 | 64.4 |
| 30 c | 0.185 | 2.797 | 108.7 |  | 0.1 | 0.189 | 2.80 | 112 |

## Gate operating point (committed in `5d48002`)

Moved locked-phase presence cut **0.3 → 0.5**. Set-difference analysis (see tooling): the estimates 0.3 admitted over 0.7 (~83 k blocks, ~7 % of output) had median |err| 4 c but **p99 140 c, 2.7 % > 50 c, ~2.2 % gross even on loud frames** — genuinely noisy. 0.5 rebalances: FNR 0.222 (vs 0.206@0.3 / 0.243@0.7), p99 39.6 (vs 59.5 / 26.0), median ~unchanged. Full presence sweep is in `eval/gate-tuning-log.md`.

User priority: **low FNR > low RMS**, but the **tail (p99) and FPR are not free** — surface them and let the user pick the operating point.

## Tooling / how to reproduce

Benchmark CLI knobs (no rebuild needed; default = shipping config):
```bash
BIN=./build/Release/saint/PitchDetector/Test/PitchDetectorImplTests
$BIN algorithm=impl thresholdWithEstimateConstraint=<x>      # locked-phase presence cut
$BIN algorithm=impl lockedConsistencyCents=<c> lockedSecondaryPeakFloor=<f>   # consistency gate (bc5e561)
$BIN algorithm=impl dumpAllBlocks=true                       # per-block CSV (bc5e561)
$BIN algorithm=impl updateBenchmarkReferences=true           # re-seed goldens after an accepted change
```
- A **failing** gate run **overwrites** the golden file with the new value (see `checkReference`); a passing run (within abs tolerance 0.01) leaves it. So after any sweep, `git checkout -- eval/BenchmarkingOutput/` to restore. Benchmark runs also append a line to `eval/cpu-realtime-log.md` — revert it too.
- `dumpAllBlocks=true` writes `eval/out/allBlocks.csv` (`id,block,weight,finalHz,errorCents,presenceScore,harmonicity`). Two runs at two gate settings emit rows in the **same corpus order**, so zip them line-by-line and set-difference the emitted truth-active rows (`finalHz>0 && weight>0`) to isolate exactly what the looser gate admits.
- **Replay dumper** (was in scratch, recreate as needed): a tiny `main` linking `RecordingReplay` + `PitchDetector` that feeds the WAV through `saint::ReplayPitchDetector::fromFile()` and prints per-block `freq` + `DebugOutput`. Build by linking `libRecordingReplay.a libPitchDetector.a libpffft.a libUtils.a -lm`. Useful debug keys exposed in `bc5e561`: `harmonicConsistencyCents`, `secondaryPeakScore` (plus `presenceScore`, `xcorrEstimate`, `probNotOctaviated`, `harmonicity`, `isOnset`, `hold`).
- Python ACF replica params (for offline analysis): MTT window 3372 samples (coeffs `{1, -1152/983, 515/2792}`, normalised), FFT 4096, `lpWindow` flat to 1500 Hz then linear roll-off 200 Hz; the analysis window for block _b_ ends at sample `(b+1)*512`.

## Open questions / next steps

1. **p90/p95 at matched FNR** — the band where the consistency gate's wander-rejection actually shows (median/p99 are blind to it). Compute from `dumpAllBlocks` (presence@0.5 vs consistency@7c, both FNR≈0.222). Decides whether consistency is worth keeping for moderate-tail accuracy.
2. **Repurpose the 2L-consistency as the band-pass trigger** (instead of the sub-fundamental energy ratio) — it's a cleaner, level-free "the estimate is being pulled" signal.
3. **Re-benchmark the band-pass on the 0.5 baseline** (its numbers above were on the old 0.3 baseline).
4. **Band-pass median cost** — try a less skirt-clipping variant (harmonic comb, or a notch that removes only the contaminant below the fundamental skirt rather than a fundamental-only band-pass).
5. **Decide keep/revert** the band-pass and consistency code (both default-OFF in `bc5e561`).
6. Rejected: _level-relative release_ (`s > α·s_lock`) — the presence score is a normalised SNR/periodicity coefficient, not a level, and the drop magnitude is set by the unknown ambient noise floor, so there's no knowable anchor. See `eval/gate-tuning-log.md` for other structural ideas.

## Resuming the session vs this doc

This Claude Code session is stored **locally** under `~/.claude/projects/…`. On the same machine you can resume it with `claude --resume` (pick from the list) or `claude --continue` (most recent in this directory). That replays the full (long) transcript. For a clean pickup, another machine, or hand-off to a fresh session, **this doc + commit `bc5e561` is the portable record** — start by reading this file and `eval/gate-tuning-log.md`, then `git branch decay-tail-wip bc5e561`.
