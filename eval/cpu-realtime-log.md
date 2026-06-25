# process() real-time CPU cost

Release, full-benchmark only. RT% = CPU time in `process()` / audio duration x 100,
measured with a per-thread CPU clock (contention-free). Machine- and load-dependent
and not reproducible, so it is recorded here for comparison rather than gated.

| date | branch | commit | message | algorithm | machine | RT% |
|------|--------|--------|---------|-----------|---------|-----|
| 2026-06-17 | master | f4f6998 | Benchmark: record per-algorithm process() real-time CPU cost (Release) | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 2.19 |
| 2026-06-17 | HEAD | 93a740b | harmonicity-gate tuning -> RMS99 reduced a lot | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 2.90 |
| 2026-06-17 | harmonic-gate | d9fd71b | Enable acf avg -> FNR looses 2% | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 3.52 |
| 2026-06-18 | master | 8e56b95 | Merge pull request #3 from saintmatthieu/harmonic-gate | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 3.54 |
| 2026-06-19 | onset-detection-improvements | ebe8ea3 | backup | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 3.52 |
| 2026-06-22 | master | 9244f4a | Remove hold from benchmarking | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 1.97 |
| 2026-06-22 | master | cb75273 | Half FFT size | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 1.01 |
| 2026-06-22 | master | fe5bb15 | Remove zero-padding completely | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 0.92 |
| 2026-06-25 | master | b3955af | README about PESTO experiment | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 0.80 |
