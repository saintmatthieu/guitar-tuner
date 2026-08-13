# Guitar Tuner — project guide for Claude

A C++17 real-time pitch-detection library for instrument tuning. The in-house algorithm
combines autocorrelation-based frequency estimation with Bayesian octave disambiguation and
harmonic fitting (see `README.md` for the algorithm). Alongside it, several third-party
detectors are wired into a shared benchmark so they can be compared head-to-head.

## Build

CMake, out-of-source. The usual build dirs are `build/{Release,Debug,Asan}`.

```bash
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release [options]
cmake --build build/Release --target <target> -j"$(nproc)"
```

Options:
- `-DNO_GTEST=ON` — skip GoogleTest and all test targets.
- `-DSAINT_WITH_AUBIO=ON` — build the aubio detectors (GPL-3.0; benchmark/TestApp only).
- `-DSAINT_WITH_PESTO=ON` — build the PESTO ONNX detector (LGPL-3.0; benchmark/TestApp only).
- `-DSAINT_WITH_PYIN=ON` — build the pYIN detector (GPL-2.0+; benchmark/TestApp only).

Each `SAINT_WITH_*` flag fetches its dependency via `FetchContent` at configure time, so the
first configure with one enabled needs network access.

## The pitch-detection benchmark (the heart of algorithm work)

The benchmark lives in `saint/PitchDetector/Test/PitchDetectorImplTests.cpp`
(`TEST(PitchDetectorImpl, benchmarking)`). It runs **one algorithm per invocation** over a
corpus of real note recordings mixed with real noise at several SNRs, scoring per 10 ms block.

### Running it

```bash
./build/Release/saint/PitchDetector/Test/PitchDetectorImplTests algorithm=<id>
# single case (writes eval/out/*_<id>.py plots, prints the CSV line):
./build/Release/.../PitchDetectorImplTests "algorithm=<id>" \
    "testCaseId=testFiles/notes/e4.wav | testFiles/noise/AC_noise.wav | -50"
# accept changed metrics as the new reference:
./build/Release/.../PitchDetectorImplTests algorithm=<id> updateBenchmarkReferences=true
```

`algorithm` defaults to `impl` (the in-house algorithm). Per-algorithm CLI knobs are read with
`getArgument<T>(...)` from the gtest argv (`T` ∈ `int|bool|std::string|fs::path`).

### Algorithms registered (`saint/PitchDetector/Test/BenchmarkAlgorithms.cpp`)

`getBenchmarkAlgorithms()` returns a `map<string, BenchmarkAlgorithm>` where
`BenchmarkAlgorithm = { factory, gates }`. Current IDs:
- `impl` — in-house (always built). Wrapped in the median filter + smoother unless
  `testWithMedianFilter=...` is set.
- `pesto` — `#ifdef SAINT_WITH_PESTO`.
- `aubio-{yin,yinfft,yinfast,mcomb,fcomb,schmitt,specacf}` — `#ifdef SAINT_WITH_AUBIO`.
- `pyin` — `#ifdef SAINT_WITH_PYIN`. Causal/streaming (fixed-lag) variant; see
  `pyin-integration.md` and `pyin-benchmark-results.md`.

### The detector contract (`saint/PitchDetector/PitchDetector.h`)

Every algorithm is a `saint::PitchDetector`:
- `float process(const float* block, DebugOutput* = nullptr, std::vector<float>* = nullptr)` —
  one block of `blockSize*numChannels` interleaved samples in; returns Hz, or `0` if no pitch.
  Write a `[0,1]` confidence into `(*debugOutput)["presenceScore"]` — the ROC/AUC reads exactly
  this key.
- `int delaySamples() const` — algorithmic latency; the harness uses it to align estimates with
  ground-truth note on/offset times (affects FPR/FNR, not the cents error).

`blockSize = sampleRate/100` (10 ms). `DebugOutput` is `unordered_map<string,float>`.

### Metrics & gating

Per-block metrics: **AVG** (mean signed cents error), **RMS** (cents), **FPR**, weighted
**FNR**, **AUC** (area under the presence-score ROC). An algorithm declares which metrics gate
it (`MetricGate`s). Reference values live in golden files
`eval/BenchmarkingOutput/<stem>[_<algo>].txt` (`RMS_error`, `FNR`, `AUC`; the default `impl`
files have no suffix). On first run a golden is **seeded**; afterwards the metric is compared
within ±1 % and a mismatch fails the gate. Re-seed with `updateBenchmarkReferences=true`.

### Corpus & outputs

- Clean notes: `eval/testFiles/notes/` (filename encodes the ground-truth pitch; `<note>.txt`
  has start/end times). Noise: `eval/testFiles/noise/`. The harness builds the Cartesian
  product notes × (noise + silence) × {−40,−50,−60 dB, silence}.
- Per-run artifacts go to `eval/out/` (git-ignored): `benchmarking[_<algo>].csv`,
  `frequencyEstimates*.py`, `presenceScores*`, `errors*.py`, `roc_curve*.py`. Plot them with the
  scripts in `eval/` (`showRoc.py`, `showFrequencyEstimates.py`, `showHistogram.py`, …).
- Test-input preparation (mixing every note with every noise at every SNR) is parallelized
  across samples in `prepareTestCases` (~2 s on a 24-thread machine). Results are placed in
  per-sample slots, so the test case order — and every metric — is identical to a sequential
  run. A disk cache of the mixes was tried and removed: regenerating in parallel is faster
  than reading the ~4.8 GB blob back.

## Adding a third-party algorithm (the `SAINT_WITH_<LIB>` template)

Mirror `saint/PitchDetector/Aubio/` (simplest analogue) or `saint/PitchDetector/Pyin/`:
1. New subdir `saint/PitchDetector/<Lib>/` with a `CMakeLists.txt` that does
   `option(SAINT_WITH_<LIB> ... OFF)`, early-`return()` if off, `FetchContent` the dependency,
   and builds a `<Lib>PitchDetector` static lib with `target_compile_definitions(... PUBLIC
   SAINT_WITH_<LIB>)`.
2. A `<Lib>PitchDetector : public PitchDetector` wrapper (down-mix to mono, buffer to the
   algorithm's frame/hop as needed, map confidence → `presenceScore`, return Hz-or-0).
3. `add_subdirectory(<Lib>)` in `saint/PitchDetector/CMakeLists.txt`; link it in
   `saint/PitchDetector/Test/CMakeLists.txt` and `saint/TestApp/CMakeLists.txt` under
   `if(SAINT_WITH_<LIB>)`.
4. Register it in `BenchmarkAlgorithms.cpp` under `#ifdef SAINT_WITH_<LIB>` with its `MetricGate`s.
5. Document it: `<lib>-integration.md` (how/why it's wired) + `<lib>-benchmark-results.md`
   (numbers). See `aubio-*.md`, `pesto-*.md`, `pyin-*.md`.

**Policy:** third-party detectors are GPL/LGPL and are **benchmark/TestApp only — never part of
the production `PitchDetector` library** (the project itself is MIT). They are fetched, not
vendored into the repo. Keep them behind their `SAINT_WITH_*` flag and out of the production
target.
