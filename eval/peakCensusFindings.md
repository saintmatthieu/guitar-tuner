# Autocorrelation peak census — findings (2026-06-12)

Question investigated: when the highest autocorrelation peak is not the fundamental,
does the fundamental still stand out as a local maximum, and do the *other* peaks'
values carry enough information to disambiguate statistically?

**Answer: yes on both counts.** The true peak survives as a local max in the large
majority of error blocks, and simple value ratios between comb peaks separate the
octave-error cases from correct picks almost perfectly.

## Method

- Collector: `saint/PitchDetector/Test/AutocorrPeakCensusTests.cpp` (target
  `AutocorrPeakCensusTests`). Runs only Preprocessor → FrequencyDomainTransformer →
  AutocorrPitchDetector, **unconstrained** (no Bayesian gate, no disambiguator, no
  onset/constraint logic), over the same test matrix as the benchmark
  (110 notes × 12 noises × 3 levels + silence = 4 033 cases). For every 10 ms block
  it dumps **all** local maxima of the ACF: quad-refined lag, raw value r(τ)/r(0),
  window-corrected value (the presence-score normalisation), and a flag for peaks
  before the first negative crossing (which the detector's search skips).
  Output: `eval/out/peakCensus_{cases,blocks,peaks}.csv` (4 033 cases, 2.5 M blocks,
  13.3 M peaks, ~40 s runtime). Accepts the benchmark's `testCaseId=` argument.
- Analysis: `eval/peakCensus.py` → `eval/out/peakCensus_report.txt` and a per-block
  modeling table `eval/out/peakCensus_blockSummary.csv.gz`.
- Each peak is classified by its lag ratio to the ground-truth lag T into classes
  {T/4, T/3, T/2, 2T/3, 3T/4, T, 4T/3, 3T/2, 5T/3, 2T, 5T/2, 3T, …} within a
  tolerance (see "Class tolerance" below; default ±25 cents), else "other".
- Headline stratum: tOnset ∈ [0.1 s, 1 s] (the window where the unconstrained stage
  operates in production). Noise levels are noise RMS with the clean note at −10 dB
  peak, so **−40 is the noisiest** condition and −inf is clean.

## Class tolerance: why ±25 cents (not 50, not 5)

`eval/peakCensusTolerance.py` measures how far **true** comb peaks sit from their
nominal lag in clean/−60 head-stratum blocks (nearest peak within ±100 ¢ of each
class position):

| class T deviation (cents) | q0.5% | q5% | med | q95% | q99.5% |
|---|---|---|---|---|---|
| total                | −22.6 | −10.9 | −1.2 | +3.8 | +15.7 |
| per-note offsets     | −18.2 | −8.4  | −1.1 | +2.9 | +9.7  |
| within-note residual | −11.9 | −3.8  |  0.0 | +2.5 | +7.6  |
| low register (<110 Hz), total | −26.7 | −18.5 | −4.9 | +2.9 | +17.8 |

- **±5 ¢ would be far too tight**: >5 % of true T peaks deviate by more than 10 ¢
  *flat*; in the low register the *median* true peak is −4.9 ¢ off. The ACF peak
  position aggregates all partials, including strongly stretched high ones, so
  inharmonicity + per-note tuning offsets + parabolic-refinement bias add up to much
  more than the few cents one would expect from low-order harmonics alone. The skew
  is systematically negative (shorter lag = sharper) — a fingerprint of string
  stretch, and potentially a model feature in its own right.
- **±50 ¢ was indeed over-crediting noise** (the original concern): tightening to 25
  halved the apparent T/3 occurrence (12.8 % → 7.7 % of blocks — that difference was
  noise wiggles being credited as comb peaks) and raised the comb-class median value
  from 0.84 to 0.93. The key separability results (B/C below) are insensitive to the
  tolerance — the structure is real, not an artefact of loose binning.
- ±25 covers ~99 % of true T peaks including the low register (whose q0.5 % = −26.7
  is right at the edge — if anything, loosen to 30, not tighten). Configurable via
  `peakCensus.py --tolerance`.

All numbers below are at ±25 ¢ (the ±50 run is kept in
`eval/out/peakCensus_report_tol50.txt`).

### Peak-relative tolerance (what the algorithm will use)

The ±25 ¢ above is for *labeling against the ground truth*, which bakes in the
per-note tuning offset. The algorithm anchors on a **found** peak at lag L and
searches for partners at 2L, L/2, … — there the systematic offsets cancel.
Measured deviation of the nearest peak to ratio × anchorLag (anchor = refined
T peak, clean/−60 head-stratum blocks):

| partner | q0.5% | q5% | med | q95% | q99.5% |
|---|---|---|---|---|---|
| 2L | −3.3 | −0.3 | 0.0 | +0.2 | +2.3 |
| 3L | −6.4 | −0.6 | 0.0 | +0.2 | +2.8 |
| L/2 (all) | −87 | −15 | +0.4 | +40 | +90 |
| L/2 (high register) | −80 | −10 | −0.2 | +2.5 | +56 |

- **Integer multiples are extremely tight: a ±5 ¢ pairwise window captures ~99 %**
  of genuine 2L/3L partners (median alignment is sub-cent). The ACF repeats at
  multiples of the *actual* peak lag, so tuning offset and inharmonicity largely
  cancel.
- **Half-lag (L/2) partners are genuinely displaced** by tens of cents, not just
  mislabeled noise: at L/2 the even-harmonic comb sits on top of the *negative*
  odd-harmonic dip, and that interference shifts the local maximum (worst in the
  low register). The truth-relative T/2 spread (±20–70 ¢ residual) confirms this
  is physical, not a tuning artefact.
- **Design implication**: verify comb membership *downward* (integer multiples of
  the shorter-lag candidate) with a tight ±5 ¢ window, rather than searching L/2
  with a loose one. To test "is the max at L an octave-down error?", anchor on the
  L/2-region peak and check that its 2×, 3× partners align within ±5 ¢ — every
  pairwise check then uses the tight integer-multiple geometry.

## A. Census: where does the unconstrained max land? (% of head-stratum blocks)

| noise | T     | T/2  | T/3  | 2T   | 3T   | other |
|-------|-------|------|------|------|------|-------|
| clean | 97.63 | 1.10 | 0.07 | 0.53 | 0.04 | 0.6   |
| −60   | 97.33 | 1.12 | 0.07 | 0.69 | 0.09 | 0.7   |
| −50   | 95.43 | 1.18 | 0.07 | 1.44 | 0.30 | 1.5   |
| −40   | 89.55 | 1.31 | 0.10 | 3.16 | 0.96 | 4.8   |

- 3rd harmonic (T/3) max: happens, but rare (~0.1 %). T/2 is 10–15× more common.
- Subharmonic (2T) max: **the dominant error in noise** (3.2 % at −40).
- By register: low notes (<110 Hz) get exclusively T/2 errors (3.5 %; 2T is outside
  the search range), mid/high notes get 2T/3T errors. Error type is structurally
  register-dependent.
- 3T errors (estimate = f₀/3, "octave+fifth too low") reach ~1 % in noise. Fixing
  them needs a ×3 candidate, which the current disambiguator does **not** have
  (its ÷3 candidate fixes T/3 errors, which barely occur). Likely a real gap.
- Ranking by *window-corrected* value instead of raw would be catastrophic
  (2T would win 14–20 % of blocks): the correction removes the lag taper that
  naturally protects T against the near-tie r(2T) ≈ r(T). Raw value must remain the
  selection feature; corrected values/ratios are for scoring and disambiguation.

## B. Ceiling: is the true peak still there when the max is wrong?

| noise | T-peak present (all blocks) | max≠T rate | T-peak present in error blocks |
|-------|------------------------------|------------|--------------------------------|
| clean | 99.57 %                      | 2.37 %     | 81.97 %                        |
| −60   | 99.56 %                      | 2.67 %     | 83.71 %                        |
| −50   | 99.17 %                      | 4.60 %     | 81.93 %                        |
| −40   | 96.29 %                      | 10.90 %    | 65.94 %                        |

Peak-based disambiguation can recover the large majority of errors; the remainder
(no T peak within ±25 ¢) needs spectral evidence — or sits 25–50 ¢ off (at ±50 ¢
tolerance the error-block presence reads 76–99 %; the truth is in between, since
weak true peaks get displaced by interference). The `wentNegative` rule hides the
T peak in ≤9 % of the recoverable error blocks — a model should also see
pre-crossing peaks. When the max is wrong, the T peak ranks top-4 by corrected
value in ~92 % of blocks (1: 36 %, 2: 23 %, 3: 18 %, 4: 15 %).

## C. The key result: relational features separate the errors

Distribution of value ratios (corrected values, blocks where both peaks exist):

| feature      | correct blocks (max=T)        | error blocks                    |
|--------------|-------------------------------|---------------------------------|
| v(T/2)/v(T)  | med 0.40, **q95 = 0.82**      | **q05 = 0.85**, med 0.99 (max=T/2) |
| v(2T)/v(T)   | med 0.999, **q95 = 1.019**    | **q05 = 1.052**, med 1.24 (max=2T) |

Nearly disjoint: a single threshold on the half-lag (resp. double-lag) value ratio
separates octave-up (resp. octave-down) errors at ~95 % on both sides, before any
model is fit. This is the cross-peak correlation that motivated the census, and it
is insensitive to the class tolerance.

Additional structure:
- Comb peaks (median corrected value 0.93) vs non-comb peaks (median 0.27) are well
  separated — spurious noise peaks are recognizable, as hypothesised.
- Caveat: pre-onset (noise-only) blocks produce a peak above 0.8 in 2–3 % of cases —
  the TV/radio and voices noises are themselves pitched. Rare but real; matters for
  the gate, not for disambiguation.

## D. Strata (validates focusing on [0.1 s, 1 s])

max=T rate, all noise levels pooled: attack [0, 0.1 s): 75.2 % | head [0.1 s, 1 s]:
94.2 % | tail (>1 s): 66.5 %.

## Caveats

- Within-note correlation: ~90 blocks per note×noise cell are nearly identical;
  effective sample size is much smaller than row counts. Any model must be
  validated with per-recording (better: per-instrument) splits.
- Errors are broad, not concentrated: at −40, 82 % of note×noise cells contain at
  least one error block in the head stratum.

## Suggested next step

Per-peak probability model (logistic to start): features raw value, v(T/2)/v(T),
v(2T)/v(T), comb support count, absolute lag (range awareness), possibly signed
lag deviation (inharmonicity skew). Train/evaluate on
`peakCensus_blockSummary.csv.gz` with per-recording splits; compare per-block
argmax accuracy against the A-table baselines above.

## To reproduce

```
cmake --build build/Release --target AutocorrPeakCensusTests
./build/Release/saint/PitchDetector/Test/AutocorrPeakCensusTests   # writes eval/out/peakCensus_*.csv
eval/.venv/bin/python eval/peakCensus.py [--tolerance 25]          # prints report
eval/.venv/bin/python eval/peakCensusTolerance.py                  # tolerance measurement
```
