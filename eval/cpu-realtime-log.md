# process() real-time CPU cost

Release, full-benchmark only. RT% = CPU time in `process()` / audio duration x 100,
measured with a per-thread CPU clock (contention-free). Machine- and load-dependent
and not reproducible, so it is recorded here for comparison rather than gated.

| date | branch | commit | message | algorithm | machine | RT% |
|------|--------|--------|---------|-----------|---------|-----|
| 2026-06-17 | master | f4f6998 | Benchmark: record per-algorithm process() real-time CPU cost (Release) | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 2.19 |
| 2026-06-17 | HEAD | 93a740b | harmonicity-gate tuning -> RMS99 reduced a lot | impl | 12th Gen Intel(R) Core(TM) i7-12800HX (24 threads) | 2.90 |
