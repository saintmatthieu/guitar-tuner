# aubio pitch detectors on the SAINT benchmark

Results of running every [aubio](https://github.com/aubio/aubio) pitch-detection method
through the SAINT benchmark (`TEST(PitchDetectorImpl, benchmarking)`), same corpus, noise
mixes and metrics as the in-house algorithm, so the numbers are directly comparable. Run on
2026-06-14.

## Setup

- Each method is registered as `algorithm=aubio-<method>`: `yin`, `yinfft`, `yinfast`,
  `mcomb`, `fcomb`, `schmitt`, `specacf`.
- Block (hop) size = `sampleRate/100` = 441 samples @ 44.1 kHz (10 ms), same as the in-house
  run. aubio analysis window `bufSize` = next power of two ≥ 4·blockSize, floored at 2048 → **2048**.
- Output unit `Hz`; `aubio_pitch_get_confidence` is exposed as the `presenceScore` that feeds
  the ROC/AUC. `delaySamples()` = bufSize/2.
- Metrics: **AVG** = mean *signed* cents error, **RMS** = RMS cents error (both over every
  block where a pitch was emitted, vs the file's true pitch), **FPR** = false-positive rate on
  no-note blocks, **FNR** = weighted false-negative rate, **AUC** = area under the
  presence-score ROC. Runs are deterministic.

## Results

"Ungated" = raw method output (no confidence gate). "Gated" = with the per-method confidence
threshold at the ROC's 1 %-FPR operating point applied (`confidence < threshold → 0 Hz`).
AUC is unaffected by the gate (the presence score is the raw confidence either way).

| method | AUC | thr@1%FPR | AVG (ung.) | RMS (ung.) | FPR (ung.) | FNR (ung.) | → AVG (gated) | RMS (gated) | FPR (gated) | FNR (gated) |
|---|---|---|---|---|---|---|---|---|---|---|
| **in-house** | **0.870** | 0.873 | **2.0** | **7.1** | **0.005** | **0.283** | — | — | — | — |
| yin      | 0.855 | 0.957 |   11.3 | 1498 | 0.427 | 0.060 |  100.3 | 207  | 0.005 | 0.485 |
| yinfast  | 0.855 | 0.957 |   11.3 | 1498 | 0.427 | 0.060 |  100.3 | 207  | 0.005 | 0.485 |
| yinfft   | 0.319 | 0.788 | 1014.6 | 1697 | 0.368 | 0.482 | 8591   | 8592 | 0.004 | 0.9997 |
| mcomb    | 0.449 | 0     |   45.4 |  977 | 0.426 | 0.060 | (ungated) | | | |
| fcomb    | 0.449 | 0     |  350.8 | 1423 | 0.427 | 0.060 | (ungated) | | | |
| schmitt  | 0.449 | 0     |   38.2 |  876 | 0.336 | 0.087 | (ungated) | | | |
| specacf  | 0.449 | 0.85  |  314.4 | 1530 | 0.427 | 0.060 | (unchanged) | | | |

(All cents figures rounded. In-house reference: AVG 2.02, RMS 7.09, FPR 0.0048, FNR 0.283,
AUC 0.870.)

## What we discovered

### 1. None of the aubio methods is remotely competitive with the in-house algorithm
The best aubio RMS is **207 cents** (yin/yinfast, gated) versus the in-house **7 cents** — a
~30× gap, and ~200× ungated. A tuner needs single-digit-cent accuracy; no aubio method gets
within two orders of magnitude. aubio's methods are general-purpose monophonic detectors, not
specialised for guitar's low strings and tight accuracy budget.

### 2. yin and yinfast are the same algorithm, and the best of the bunch
Their numbers are identical to rounding (yinfast is YIN computed in the spectral domain). They
have the only genuinely useful confidence: **AUC 0.855**, just below the in-house 0.870. yinfast
is also by far the fastest method — ~7 s of benchmark wall-clock versus ~53 s for the in-house
algorithm (the in-house pipeline runs a 16384-point FFT, ~3 transforms/frame, 4× autocorrelation
upsampling and a disambiguation/median/smoothing chain; yinfast is a single 2048-point spectral
YIN). So aubio buys ~8× speed for ~30–200× worse accuracy — a clean accuracy-for-compute trade.

### 3. yinfft — aubio's default — is the worst, and its confidence is *anti*-correlated
Counterintuitively, yinfft (aubio's recommended default) scores AVG 1014 cents and **AUC 0.319**.
AUC < 0.5 means *higher* confidence predicts *less* likely voiced — the score is inverted. The
large signed AVG (≈ +1000 cents, not a near-zero like yin) points to a *systematic* error
(consistent octave/harmonic misfire on guitar) rather than the random-sign noise the other
methods show. Applying its 1 %-FPR threshold therefore destroys it: it removes almost all true
positives (FNR 0.9997). A 1 %-FPR confidence gate only makes sense when AUC > 0.5.

### 4. The confidence gate fixes FPR but not accuracy, at a steep FNR cost
For yin/yinfast the gate does exactly what it's designed to: **FPR collapses 0.43 → 0.005**. But:
- **FNR jumps 0.06 → 0.49** — it now misses half the notes.
- **RMS only drops 1498 → 207** and AVG *rises* 11 → 100.

The reason: aubio's confidence does **not** separate octave errors from correct detections. The
gross errors on the low strings are *confident-but-wrong*, so they survive the gate while many
correct (quieter) detections are cut. Gating improves "is a note present?" but barely touches
"what note is it?".

### 5. RMS ≫ AVG everywhere — a heavy-tailed, sign-symmetric error distribution
yin ungated has AVG 11 cents but RMS 1498. That isn't a contradiction: most detections are
accurate (AVG, the *signed* mean, stays near zero because the big +/− errors cancel), but a
minority are catastrophic (octave errors + noise-driven garbage), and RMS squares them so they
dominate. The worst cases are all **low strings at high noise** — e.g. yin's worst is 9914 cents
on `E2.wav` mixed with `Gibson_LP_noise` at −40 dB. This is YIN's classic period-doubling
failure mode, worst where it hurts a guitar tuner most.

### 6. mcomb / fcomb / schmitt / specacf have no usable confidence → can't be gated
aubio only registers a confidence callback for yin, yinfft, yinfast (and specacf via its
tolerance). mcomb/fcomb/schmitt return a constant, so their ROC is degenerate (the identical
AUC 0.449 just reflects the label ordering, not discrimination) and the 1 %-FPR threshold comes
out **0** — i.e. ungated. specacf's confidence is effectively constant ≥ 0.85, so its threshold
(0.85) never fires and it stays ungated too. Their accuracy is poor regardless (RMS 876–1530).

### 7. Giving yinfast the in-house window makes voicing better but pitch worse
The in-house algorithm resolves the low E with a long, low-frequency-resolving window (a
16384-point FFT). Does yinfast improve if given the same? Sweeping its analysis window
`bufSize` (ungated, to isolate the raw effect):

| bufSize | window | AVG | RMS | FPR | FNR | AUC |
|---|---|---|---|---|---|---|
| 2048 (baseline) | ~46 ms |  +11 | 1498 | 0.427 | 0.060 | 0.855 |
| 4096            | ~93 ms | −160 | 1733 | 0.433 | 0.060 | 0.864 |
| 8192            | ~186 ms | −290 | 2028 | 0.447 | 0.062 | 0.871 |
| 16384 (= in-house FFT) | ~371 ms | −382 | 2329 | 0.472 | 0.066 | **0.876** |

Two opposite trends as the window grows:

- **Accuracy gets *worse*** — RMS climbs 1498 → 2329, and the AVG drifts increasingly *negative*
  (+11 → −382 cents). A negative signed mean means estimates are too *low*: octave-**down**
  (subharmonic) errors. A longer YIN window has more deep minima in its difference function at
  integer multiples of the true period, so it locks onto a sub-multiple more often. More data
  makes plain YIN's pitch *less* reliable, not more. (The worst case stays ~8800 cents on E2.)
- **Confidence gets *better*** — AUC rises monotonically 0.855 → 0.876, edging *past* the
  in-house 0.870 at bufSize 16384. A longer window gives a cleaner, more stable confidence, so
  the "is a note present?" decision improves even as "which note?" degrades.

The takeaway: the in-house algorithm's accuracy does **not** come from its window length —
yinfast has the same long window here and gets *more* octave errors. The in-house edge comes
from the **`AutocorrEstimateDisambiguator`** stacked on top, the stage that resolves exactly
these octave/subharmonic ambiguities. Window length buys voicing confidence; disambiguation
buys cents accuracy, and aubio's YIN has none.

## Conclusion

The integration is healthy and the methods perform at roughly their published general-purpose
level — but that level is far from what a guitar tuner needs. **yin/yinfast** are the only
methods with a meaningful confidence (AUC 0.855) and are the fastest, but even gated their RMS
of 207 cents rules them out for tuning. **yinfft**, despite being aubio's default, is the worst
and has an inverted confidence on this corpus. The in-house algorithm wins decisively on
accuracy, at ~8× the compute. If aubio were ever to be used, yinfast is the only candidate, and
only as a cheap coarse/voicing front-end — never as the pitch source. Widening its window to
match the in-house FFT pushes its voicing AUC past the in-house algorithm's (0.876 vs 0.870)
while *worsening* its pitch (RMS 2329), which only reinforces that split.
