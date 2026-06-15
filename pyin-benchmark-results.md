# pYIN on the SAINT benchmark

Result of running pYIN (`algorithm=pyin`) through the SAINT benchmark
(`TEST(PitchDetectorImpl, benchmarking)`) — same corpus, noise mixes and metrics as the in-house
algorithm, so the numbers are directly comparable. Run on 2026-06-15.

See `pyin-integration.md` for how pYIN is wired. **This is the causal (fixed-lag) variant** — what
a real-time tuner could run — not pYIN's offline/published accuracy (see the caveat below).

## Setup

- `algorithm=pyin`, defaults: `pyinFrameSize=2048`, `pyinFixedLag=20`, `pyinThreshDistr=2`
  (Beta mean 0.15), `pyinLowAmp=0.1`.
- Block (hop) = `sampleRate/100` = 441 samples @ 44.1 kHz (10 ms), same as every other run.
- `presenceScore` = the YIN voiced-probability mass (feeds the ROC/AUC).
  `delaySamples() = frameSize/2 + (fixedLag-1)·blockSize` ≈ 213 ms @ 44.1 kHz.
- Benchmarked raw (no in-house median filter / smoother — pYIN does its own HMM smoothing).
- Metrics: **AVG** = mean signed cents error, **RMS** = RMS cents error (over blocks where a
  pitch was emitted), **FPR** = false-positive rate, **FNR** = weighted false-negative rate,
  **AUC** = area under the presence-score ROC. Deterministic. Corpus: 109 clean notes × noise ×
  SNR = 4033 test cases.

## Results

| algorithm | AVG (cents) | RMS (cents) | FPR | FNR (weighted) | AUC |
|---|---|---|---|---|---|
| **in-house (`impl`)** | 2.0 | **7.1** | 0.005 | 0.283 | 0.871 |
| `aubio-yin` (gated) | 100.3 | 207 | 0.005 | 0.485 | 0.855 |
| **`pyin` (causal)** | −145.5 | 262 | **0.0012** | **0.240** | **0.875** |

(pYIN gated by its own voicing; worst single case 2759 cents.)

## Interpretation

- **Best voicing detection of the three.** pYIN has the highest AUC (0.875) and the lowest FNR
  (0.240) — its HMM voicing model decides "pitch present?" better than `impl` or `aubio-yin`, and
  with a very low false-positive rate (0.0012).
- **Poor pitch accuracy, by octave/subharmonic error.** RMS 262 cents and a strongly negative AVG
  (−145) mean the causal decode systematically locks onto subharmonics (T/2, T/3) on sustained
  notes — the same failure mode as plain `aubio-yin` (RMS 207), and far worse than `impl` (7.1).
- **Net:** as a *real-time* detector on this guitar corpus, pYIN is not competitive with the
  in-house algorithm on accuracy, though its voicing/confidence is excellent.

## Caveat: this is the causal ceiling, not pYIN's offline accuracy

pYIN's published strength comes from **offline** Viterbi over the whole signal. Verified
separately: the *same* vendored core, decoded offline, reproduces librosa's pYIN (e.g. `e4.wav`
→ 329 Hz, correct). The causal fixed-lag decode used here cannot do that for sustained notes
(per-frame candidates are near-equal across subharmonics; only full-file Viterbi disambiguates),
so it octave-errors (`e4.wav` → 109 Hz). No `pyinFixedLag` value fixes it (swept to 200).

If an *offline* pYIN comparison is ever wanted (pYIN's algorithmic ceiling, comparable to the
paper/librosa), it would need a benchmark path that feeds the whole file and decodes once at the
end — a harness change, out of scope for this causal/real-time evaluation.
