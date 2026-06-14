# PESTO vs. SAINT — pitch-detector benchmark results

Date: 2026-06-12. Follow-up to `pesto-integration.md`. Both algorithms were run over the
identical corpus (4033 test cases: 109 labelled notes × {silence, 12 noises × 3 levels}),
same metrics, same machine, via `TEST(PitchDetectorImpl, benchmarking)`.

- **SAINT** = in-house algorithm as shipped (`algorithm=impl`, median filter + smoother).
- **PESTO** = `mir-1k_g7` checkpoint (the repo's recommended default), streaming ONNX export,
  run **raw** (no SAINT post-processing), confidence threshold 0.5.

## Headline numbers

| Metric                | SAINT (impl) | PESTO (raw) |
|-----------------------|-------------:|------------:|
| AVG error (cents)     |         2.02 |       522.5 |
| RMS error (cents)     |         7.09 |       636.1 |
| FPR (@ own threshold) |       0.0048 |      0.0029 |
| weighted FNR          |        0.283 |       0.603 |
| AUC                   |        0.870 |       0.809 |
| TPR @ 1% FPR          |        0.574 |       0.467 |
| Benchmark wall time   |         52 s |      26 min |

SAINT's run reproduced the stored baselines exactly (RMS 7.0896, AUC 0.8703, FNR 0.2833),
so the comparison is against a healthy reference.

## Where PESTO loses: sustained octave errors on low strings

Per-block error distribution over voiced blocks (estimate returned):

|                                  | SAINT | PESTO |
|----------------------------------|------:|------:|
| within ±20 cents                 | 98.3% | 58.7% |
| within ±50 cents                 | 99.7% | 66.2% |
| octave errors (≈ ±1200 cents)    |  0.0% | 18.7% |
| other gross errors (> 50 cents)  |  0.2% | 15.1% |

Breakdown by note (PESTO, median per-case RMS in cents):

| note | E2 (82 Hz) | A2 (110 Hz) | D3 (147 Hz) | G3 (196 Hz) | B3 (247 Hz) | E4 (330 Hz) |
|------|-----------:|------------:|------------:|------------:|------------:|------------:|
| median RMS | 1404 | 808 | 162 | 21 | 7.6 | 2.3 |

Two key observations:

1. **The errors are noise-independent.** Average RMS is ~570 cents even on the clean
   (-inf noise) mixes — this is not a robustness failure, it's a domain mismatch. Verified
   directly: a clean `guitar_1/e2.wav` through the model locks onto exactly +1200 cents
   (164.8 Hz, the 2nd harmonic) for the entire note. `mir-1k_g7` was trained on MIR-1K
   (singing voice), and low guitar strings are outside its comfortable range — even a pure
   110 Hz sine shows a +41-cent bias (440 Hz is dead-on).
2. **When PESTO is right, it is extremely precise**: median |error| of the within-±50-cents
   blocks is ~0 cents (SAINT: 2.1 cents). On E4/B3 it is excellent. The failure mode is
   discrete harmonic/octave confusion, not estimation noise — so wrapping it in the
   median filter/smoother would *not* help (the octave errors are sustained, not transient).

## Confidence calibration

- At the paper-ish default threshold 0.5, PESTO is very conservative: FPR 0.003 but
  FNR 0.60. The ROC picks threshold **0.109** for the benchmark's 1%-FPR operating point.
- AUC 0.809 vs SAINT 0.871; at every FPR the SAINT classifier dominates
  (TPR@1%FPR 0.467 vs 0.574). PESTO's confidence also saturates at 1.0 on strong notes,
  giving the ROC little to work with in the easy region.
- Re-runnable knob: `pestoThreshold=<float>` test argument.

## Latency and cost

- Models exported with step = 10 ms (chunk 441 @ 44.1 kHz, 480 @ 48 kHz) matching the
  benchmark's block size, so the wrapper adds **no FIFO buffering**; `delaySamples()`
  reports one block (10 ms). The model's internal analysis window adds unknown latency on
  top — if anything, that misalignment slightly inflates PESTO's FPR/FNR, but nowhere near
  enough to change the verdict.
- CPU: **1.28 ms per 10 ms frame** (single-threaded x86 CPU, ONNX Runtime 1.26) ≈ 13% of
  real time — vs. the repo's quoted 0.7 ms/frame at the coarser 23 ms step. Whole-benchmark
  wall time 26 min vs SAINT's 52 s. Model file: 17 MB per sample rate.

## Verdict

Raw PESTO (`mir-1k_g7`) loses decisively on this corpus: ~34% of its voiced estimates are
gross errors, dominated by octave jumps on E2/A2 — precisely the guitar-tuner-critical low
strings — and its detector ROC is uniformly below SAINT's. The integration itself is
validated (Python and C++ paths agree to the block level; sine tones convert exactly), so
the gap is the model, not the plumbing. If PESTO stays interesting, the path is the one
already anticipated in `pesto-integration.md`: **self-trained weights on guitar material**
(and ideally a pitch range extended below 80 Hz), then re-run this same benchmark — that
would also resolve the LGPL question for deployment.

## Validation: reproducing the published MIR-1K numbers (2026-06-12)

To rule out an integration/methodology bug behind the poor guitar numbers, the paper's own
benchmark (ISMIR 2023, Table 1: RPA = % of voiced frames within 50 cents, MIR-1K clean
vocal channel, mir_eval) was reproduced on the full 1000-clip MIR-1K dataset:

| Evaluation | RPA (frame-weighted) |
|---|---:|
| Published (v1 `mir-1k` checkpoint, paper Table 1) | 96.1% |
| Our reproduction (v1 checkpoint, pesto-pitch 1.0.0, batch) | **96.0%** |
| `mir-1k_g7` (the checkpoint we integrate), batch | 98.3% |
| `mir-1k_g7`, streaming ONNX 16 kHz/10 ms (= our C++ wrapper path) | 94.8% |

- The published number reproduces to within 0.1 points → the evaluation pipeline
  (channel selection, label alignment, metric) is sound.
- The g7 checkpoint is even stronger than the published model on its home domain.
- The streaming export — the exact path our C++ wrapper uses — gives up ~3.5 points vs.
  batch (causal cache vs. centered frames + zero-cache warm-up on 4–13 s clips), but is
  still ~95%. The same path scores ~66% (within ±50 cents) on the guitar corpus.

Conclusion: the integration is healthy; PESTO genuinely performs at its published level on
singing voice and genuinely fails on low guitar strings. Scripts:
`/home/matthieu/github/pesto-work/eval_mir1k{,_v1,_onnx}.py` (dataset in
`pesto-work/mir1k/`; the v1 checkpoint needs `pesto-pitch==1.0.0` in `venv1` — the current
code can't load it).

## How to reproduce

```sh
# one-time: export models (writes into saint/PitchDetector/Pesto/models/, gitignored)
/home/matthieu/github/pesto-work/export_models.sh

cmake --preset Release -DSAINT_WITH_PESTO=ON
cmake --build build/Release --target PitchDetectorImplTests
./build/Release/saint/PitchDetector/Test/PitchDetectorImplTests \
    --gtest_filter='*benchmarking*' algorithm=pesto [pestoThreshold=0.11]
```

The wrapper lives in `saint/PitchDetector/Pesto/` (static lib, never linked into the
production library). The TestApp can run PESTO live: `TestApp --pesto` (same build flag;
uses a dedicated 44.1 kHz / 512-sample export and threshold 0.11).

Outputs land in `eval/out/*_pesto.{csv,py}` (CSV, errors, ROC) alongside the default
algorithm's files. Note: pesto's `realtime/export_onnx.py` needs `dynamo=False` patched
into its `torch.onnx.export` call on torch ≥ 2.9 (the dynamo exporter fails on
`aten::roll`); the local clone at `/home/matthieu/github/pesto-work/pesto` has this patch.
