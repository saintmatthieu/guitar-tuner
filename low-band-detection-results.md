# Below-range detection — measurements

Two attempts at the same question. The second one ships:

1. **The in-range window's own brick-walled low band** — does not work at that window length.
   The four findings below are why, and they are what the second attempt is built on.
2. **`LowBandAnalyzer`: a dedicated, further-decimated, long-window analysis** — works.
   Bucket error rate **0.0118 → 0.00056** with the median RMS, 99th-percentile RMS and FPR
   *unchanged to the last digit*. Jump to [the prototype](#the-prototype-that-works).

## The problem it targets

Since `5d7eb49` (*Add first out-of-range samples*) the corpus contains six A1 recordings
(54.7 Hz), below the standard-tuning search range (`getMinFreq(Standard) = 69.3 Hz`). The
in-range path locks onto an in-range **harmonic** of such a string and reports it as a
confident in-range pitch, which is what the bucket error rate scores:

| metric | before A1 samples | after (master) |
| --- | --- | --- |
| bucket error rate | 0 | 0.0118 |
| weighted FNR | 0.2224 | 0.3300 |

## Attempt 1 — the in-range window's brick-walled low band

Per frame, after the in-range estimate has passed the octaviation gate:

1. **Brick-wall low-pass** the same spectrum at the search range's lower edge and autocorrelate
   what is left. An in-range note has next to no energy there; a below-range string still has
   its fundamental. No new
   forward FFT — the spectrum is shared with the in-range path; one extra inverse FFT, and
   only for estimates low enough to have a sub-harmonic in the band (nothing above ~277 Hz).
2. **Probe the candidate periods** `k * lag`, k = 2…4, for the sub-harmonics that land in the
   band, scoring each by a *periodicity contrast*: the overlap-corrected autocorrelation at
   the period minus the one half a period away. The contrast is what makes rumble - which is
   steady across the window and therefore correlates at every lag - score 0 instead of 1.
3. **Corroborate in the spectrum**: does the hypothesised fundamental fill in the harmonics the
   in-range estimate leaves empty (for k = 2, the odd multiples)? Measured as prominence at the
   comb positions of the whitened spectrum, off-comb as a fraction of the total.
4. If both clear their thresholds, return `estimate / k` tagged `PitchBucket::belowRange`.

Reading the autocorrelation *at* the candidate periods, rather than picking the band's own
highest peak, keeps the reported frequency as accurate as the in-range estimate — the brick
wall truncates the analysis window's main lobe, so a frequency read off the band itself is
biased.

### What it does on the corpus

Best operating point found, against master:

| metric | master | low band on | |
| --- | --- | --- | --- |
| bucket error rate | 0.0118 | 0.0121 | no better |
| weighted FNR | 0.3300 | 0.3300 | unchanged |
| median RMS (cents) | 2.524 | 2.549 | slightly worse |
| 99th-pct RMS (cents) | 36.8 | 85.2 | **much worse** |
| `process()` real-time cost | 0.57 % | 0.64 % | +0.07 pp |

The tail is the tell: every false below-range verdict on an in-range note reports
`estimate / k`, i.e. an error of 1200 c (k = 2) or 2400 c (k = 4).

### Why — four findings, each measured

1. **Mains hum owns the low band.** Most corpus noises carry 50 Hz hum. At a 76 ms window
   (1686 samples at the decimated 22.05 kHz, `fftSize = 2048`, 10.8 Hz bins) 50 Hz and the
   A1's 54.7 Hz are **0.4 bins apart** — unresolvable. The band's highest peak is the hum, not
   the string: for the A1 recordings it lands at 49.5–49.9 Hz, so an integer-ratio test
   against the in-range estimate fails on the very cases it is meant to catch. (This is why
   step 2 probes candidate periods instead of peak-picking.)

2. **Hum also fakes the ratio for mid-range notes.** D3 = 146.8 Hz has 146.8/3 = 48.9 and
   G3 = 196 Hz has 196/4 = 49.0 — both within ~2 % of the hum. Before the spectral
   corroboration was added, 188 in-range cases misfired, D3 and G3 dominating.

3. **The wall's edge rings, and the in-range fundamental leaks through it.** The window's main
   lobe is ±39 Hz wide while the wall sits at 69.3 Hz, so an in-range E2 (82.4 Hz) keeps a
   substantial part of its main lobe in the band, and the sharp spectral edge makes it ring in
   the lag domain with a slow 1/τ decay. Measured contrast for a genuine E2 at its mirage
   period: **1.3–1.5**, *higher* than the true A1 at its correct period (1.1) — and above the
   1.0 a real signal cannot exceed, which is the artifact showing itself.

4. **The in-range estimate is not always an integer harmonic.** A1_1 mostly locks **82 Hz**,
   not 109 Hz: 164 Hz (the 3rd harmonic) is the strongest partial in that recording, and the
   autocorrelation settles on half of it. The true fundamental is then 2/3 of the estimate, so
   no integer k describes it. The premise "the in-range estimate is a harmonic of the
   below-range fundamental" does not hold for this corpus.

And the discriminators do not separate the classes. Per-frame comb support at the
*correct* hypothesis for A1_1 (median 0.04, p75 0.17, max 0.29) versus a genuine in-range E2
at a false hypothesis (median 0.05, p75 0.19, max 0.36): fully overlapping. Recomputing the
same statistic offline on the raw spectrum, with both the shipping minimum-3-term window and a
narrower-main-lobe Hann, gives 0.30–0.68 for true below-range frames and 0.41–0.61 for false
ones — so this is not a window-choice artifact.

### Worth knowing before tuning against this metric

Five of the six A1 recordings are barely
detected at all: mean per-case FNR over the 222 A1 cases is 0.95; A1_1 is detected ~27 % of
the time (and always with the wrong bucket), A1_2…A1_6 under 5 %. Nearly the whole 0.0118
comes from A1_1's loud early blocks. Since the bucket error rate only scores blocks where both
the returned and the expected bucket are pitched, *suppressing* output on those blocks reduces
the metric as effectively as classifying them correctly — an earlier iteration here scored
0.0094 that way, while pushing those cases' FNR from 0.77 to 1.00. Read the two metrics
together.

## The prototype that works

The window was the binding constraint, so the second attempt gives the question its own
window. `LowBandAnalyzer` low-passes and decimates the preprocessor's output again, and runs a
~170 ms window over it — 2.3× the in-range one, yet **fewer samples**, because the band of
interest ends a few hundred Hz up. That is Idea 2 in `low-note-detection-ideas.md` (branch
`low-note-detection-experiment`). At ~5 Hz bins a 55 Hz comb is cleanly resolved, which is what
makes the spectral evidence mean anything.

Everything about that geometry is derived from one quantity, the top of the band the comb test
reaches (`subHarmonicCombSize × ` the range floor): the anti-alias cutoff sits a fifth above it
(a Butterworth is 3 dB down at its own cutoff, so it cannot sit *on* the band), and the
decimation is then as deep as that cutoff allows (Nyquist 2.6× above it, ~50 dB of stopband for
an order-6 Butterworth). Since the range floor moves with the tuning, so does all of it: a
standard tuning decimates by 4 (5512 Hz, 1024-point FFT), a floor an octave lower by 7, and Open
G2 — whose comb reaches 989 Hz — by 3 rather than running into a fixed filter.

Note what the band is *not*: it is not the octave below the range. The evidence is the harmonic
structure across the whole comb, so the in-range energy has to be **passed**, not filtered out.
Suppressing it was attempt 1.

Per frame, once the in-range estimate has passed the octaviation gate:

1. **Candidates**: the fundamentals the estimate could be a partial of — it at small rational
   ratios q/p, q ≤ 3, p ≤ `maxHarmonic`, that land below the range. Confining the search to
   these *is* the statement that the two frequencies belong to one string; the mains hum stands
   at no such ratio to the note it accompanies.
2. **Fit**: the candidate whose harmonic comb has the most prominence in the whitened spectrum,
   every candidate weighed over the same number of harmonics.
3. **Evidence**: the share of that comb's prominence which the in-range estimate *cannot*
   explain — for an octave relation, the odd harmonics. A string really sounding below the
   range fills them in; an in-range note leaves them empty. Measured **per comb member**, not in
   total: how many harmonics the estimate explains is a property of the ratio between the two
   frequencies (every second for an octave, every third for an octave and a fifth), so a total
   scores the same physical evidence 0.5 for one ratio and 0.67 for the other, and no single
   threshold means the same thing for both. Per member, a fully present comb scores 0.5 whatever
   the ratio.
4. **Persistence**: the evidence must hold for `minConsecutiveFrames` (20 = 200 ms) before the
   verdict is issued. Withdrawing needs no delay — extending a verdict past its evidence only
   prolongs the wrong ones.

Operating point (the config defaults): `maxHarmonic = 3`, `harmonicSupportFloor = 0.35`,
`minConsecutiveFrames = 20`.

| metric | master | prototype | |
| --- | --- | --- | --- |
| bucket error rate | 0.0118 | **0.00056** | 21× lower |
| median RMS (cents) | 2.52377 | 2.52377 | identical |
| 99th-pct RMS (cents) | 36.834 | 36.834 | identical |
| FPR | 0.00358021 | 0.00358021 | identical |
| weighted FNR | 0.3300 | 0.3349 | +0.005 |
| `process()` real-time cost | 0.57 % | 0.68 % | +0.11 pp |

Only 3 of the 4033 in-range cases move by more than 20 cents (worst 95 c), and the in-range
mean bucket error and FNR are unchanged to four decimals. On the two A1 recordings the in-range
path actually fires on, the per-case bucket error drops from 1.000 to 0.089 (A1_1) and 0.297 to
0.081 (A1_2).

Three things earned their keep, each measured:

- **Bounding the harmonic order** (`maxHarmonic = 3`) is what removed the bulk of the false
  positives. The higher the in-range estimate, the more rational sub-multiples of it land in the
  band, and the likelier one lines up with the hum's harmonics rather than a string's: at
  `maxHarmonic = 12` the regressions were 331 in-range cases, led by E4 (96) and B3 (54); at 3,
  notes above ~208 Hz are not tested at all and the count fell to 77, then to 3 once the
  persistence guard was added.
- **The persistence guard** is what took the tail back to baseline: at 1 frame the
  99th-percentile RMS was 211 c, at 5 frames 48 c, at 20 frames 36.834 c — master's value
  exactly.
- **Re-locking the median filter on a bucket switch** (`PitchDetectorMedianFilter`) recovers
  what the verdict would otherwise cost. Without it the filter sees an in-range reading and a
  below-range one an octave apart inside one window, fails its own consistency check and emits
  nothing: A1_1's FNR went 0.737 → 0.883 while its bucket error fell. With it, the tail returns
  to baseline and the bucket error halves again.

### What the per-member normalisation cost on this corpus

It is the right measure and it scores slightly worse here: at the same tail (p99 36.834, the
baseline) the bucket error goes 0.00056 → 0.00117. Lowering the floor to 0.30 gives 0.00036 —
better than either — for 7 cents of tail (p99 44.2). The reason the biased version looked good is
worth stating plainly: the corpus's entire below-range population is one recording, A1_1, and it
is a p = 3 case (it reads 82 Hz, two thirds of its fundamental) — precisely the ratio the count
bias inflated, by 1.37x on the logged frame (0.63 against 0.46). On a p = 2 case, which is the
commoner one physically, the two measures are identical. So the corpus cannot say which is
better, and the biased one's edge comes from over-fitting the single example.

### Where the remaining FNR goes

The +0.005 is the price of the 200 ms of evidence plus the filter's 150 ms re-lock, which land
at the *onset* — where the FNR weight is 1. In production the hold (`defaultHoldDuration = 1 s`,
disabled in the benchmark) covers that gap, so the shipped behaviour should cost less than the
measurement shows. Rescaling the median window by the two readings' ratio instead of re-locking
was tried: 0.3328 FNR, but twice the bucket error and +4 c on the tail.

## What the corpus still cannot show

Five of the six A1 recordings are barely detected at all, and that is unchanged by any of this:
the in-range path never fires on them, so there is nothing for a below-range verdict to
reinterpret. Adding below-range material that the in-range path *does* fire on would let the
feature's benefit be measured directly, rather than through one recording (A1_1) and its
octave-locked blocks.
