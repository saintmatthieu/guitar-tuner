# Guitar Tuner — Pitch Detection Library

A C++ pitch detection library designed for real-time instrument tuning. The core algorithm combines autocorrelation-based frequency estimation with Bayesian octave disambiguation and harmonic fitting.

---

## Algorithm

Audio is processed in fixed-size blocks (10 ms). For each block, the pipeline runs the following steps:

```
Raw audio block
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              ▼
┌─────────────┐                            ┌──────────────────┐
│  Onset      │                            │  Preprocessor    │
│  Detector   │                            │  (Butterworth    │
│             │                            │  LPF @ 5 kHz)    │
└──────┬──────┘                            └────────┬─────────┘
       │ onset detected?                            │ filtered audio
       │ → reset estimate constraint                ▼
       │                               ┌────────────────────────┐
       │                               │  FrequencyDomain       │
       │                               │  Transformer           │
       │                               │  (MTT window + FFT,    │
       │                               │  zero-padded ×4)       │
       │                               └──────────┬─────────────┘
       │                                          │ complex spectrum
       │                                          ▼
       │                               ┌────────────────────────┐
       │                               │  AutocorrPitchDetector │
       └──────────────────────────────►│  (IFFT of |X|², peak   │
                                       │  search, quadratic     │
                                       │  refinement)           │
                                       └──────────┬─────────────┘
                                                  │ xcorrEstimate
                                                  │ + presenceScore
                                                  ▼
                                       ┌────────────────────────┐
                                       │  Bayesian Gate         │
                                       │  P(good | presenceScore│
                                       │  ) < threshold → 0     │
                                       └──────────┬─────────────┘
                                                  │ (passes gate)
                                                  ▼
                                       ┌────────────────────────┐
                                       │  Disambiguator         │
                                       │  (spectral whitening + │
                                       │  harmonic fit for      │
                                       │  octave selection)     │
                                       └──────────┬─────────────┘
                                                  │
                                                  ▼
                                          frequency estimate (Hz)
                                          or 0 (no pitch)
```

### Step descriptions

**Onset Detector** — Runs on the raw, unfiltered audio. Detects sudden increases in signal power using a Hann-windowed RMS with exponential smoothing. When an onset is detected (i.e. a new note attack), the estimate constraint from the previous note is cleared, allowing the algorithm to search the full frequency range.

**Preprocessor** — Applies a 6th-order Butterworth low-pass filter at 5 kHz to reject high-frequency noise before the FFT.

**FrequencyDomainTransformer** — Applies a Minimum Three-Term (MTT) cosine window to the buffered audio and computes a zero-padded (×4) real FFT via [PFFFT](https://bitbucket.org/jpommier/pffft). Samples below −60 dB are zeroed before windowing to suppress low-level noise.

**AutocorrPitchDetector** — Computes the autocorrelation function from the power spectrum (IFFT of |X|²), then searches for the dominant periodic peak. If a previous estimate is available, the search is constrained to within a major-third ratio of that estimate. Sub-sample accuracy is obtained by fitting a parabola around the peak. Outputs a frequency estimate and a *presence score* (normalised peak height).

**Bayesian Gate** — Estimates P(good | presence score) using Bayes' theorem with two fitted likelihood distributions:
- *Good* (non-octaviated) estimate: Beta(a=3.39, b=0.40)
- *Bad* (octaviated) estimate: skewed normal(a=4.58, loc=0.13, scale=0.36)
- Prior: 59% good, 41% bad

If the posterior probability falls below 0.89 (or 0.75 when a constraint is active), the block returns 0 (no pitch detected).

**Disambiguator** — Applies cepstral liftering to extract the spectral envelope and whiten the spectrum. Evaluates four candidate pitches (xcorrEstimate, ×2, ÷2, ÷3) by fitting a harmonic series to the whitened spectral peaks using iteratively re-weighted least squares. The candidate with the lowest residual error is returned.

---

## Benchmarking

The benchmarking suite lives in `saint/PitchDetector/Test/PitchDetectorImplTests.cpp`. It evaluates the full pipeline across a matrix of real recordings and synthesised noise conditions.

### Test sample construction

**Clean recordings** (`eval/testFiles/notes/`)

Each file contains a single note plucked once and allowed to ring until no pitch remains. Guidelines for recording:

- The note must be very well in tune (it is the ground truth).
- Do not mute other strings (sympathetic resonance is part of the signal).
- Aim for a clean, quiet environment — noise is added artificially by the test harness.
- The filename encodes the expected pitch; the harness extracts the ground truth frequency from it automatically.

The population of samples is intended to be representative of real-world use: if 5% of users play ukulele, ~5% of samples should be ukulele recordings. Dimensions to cover include instrument, string, recording device, and room.

**Noise recordings** (`eval/testFiles/noise/`)

Ambient noise is recorded separately (15–20 s each) and mixed in by the test harness at calibrated levels. Loudness does not need to be controlled during recording. Example noise types: air conditioner, PC fan, kitchen sounds, children, muffled TV/radio.

**Test case matrix**

The harness generates the Cartesian product of:
- All clean note recordings
- All noise recordings (plus a silence case)
- Four SNR levels: −40 dB, −50 dB, −60 dB RMS, and silence

Each combination is a separate test case. Clean audio is normalised to −10 dB peak; noise is scaled to the target RMS before mixing.

### Metrics

All metrics are computed per block (10 ms). Each block is labelled *positive* (within the known note's sustain window, pitch should be detected) or *negative* (before the attack / after the note decays, no pitch expected).

| Metric | Description |
|--------|-------------|
| **RMS error (cents)** | Root mean square of `1200 × log₂(estimate / truth)` across all positive blocks where the algorithm returned a non-zero estimate. The primary accuracy metric. |
| **False Negative Rate (FNR)** | Fraction of positive blocks where the algorithm returned 0 (missed detections). Averaged across test cases. |
| **False Positive Rate (FPR)** | Fraction of negative blocks where the algorithm returned a non-zero estimate (spurious detections). Averaged across test cases. |
| **AUC** | Area under the ROC curve built from all `(presence_score, is_positive)` pairs across the full test suite, using the trapezoid rule. Summarises the trade-off between FNR and FPR across all possible thresholds. |

The test also finds the presence-score threshold that keeps FPR ≤ 1% and reports the corresponding TPR.

**Regression thresholds** — Stored reference values are compared with ±1% tolerance on each run. A regression in any metric causes the test to fail, making the suite suitable for CI gating.

### Output artefacts

The test harness writes several files for offline analysis:

| File | Contents |
|------|----------|
| `benchmarking.csv` | Per-test-case AVG, RMS, FPR, FNR, mix ID |
| `frequencyEstimates.py` | Frequency estimates over time (for plotting) |
| `presenceScores.py` | Presence scores over time |
| `errors.py` | Per-block cent errors |
| `roc_curve.py` | ROC curve arrays and AUC |
| `*_preprocessed.wav` | Audio after the Butterworth filter, for debugging |

Scripts in `eval/` (e.g. `showRoc.py`, `showFrequencyEstimates.py`, `showHistogram.py`) can plot these outputs.
