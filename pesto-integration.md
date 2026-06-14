# Task: Integrate PESTO (ONNX) into the SAINT pitch-detector benchmark

## Goal

Evaluate the ML-based pitch detector **PESTO** against our SAINT algorithm by running it
through the existing benchmark in `saint/PitchDetector/Test/PitchDetectorImplTests.cpp`
(`TEST(PitchDetectorImpl, benchmarking)`). Same corpus, same noise mixes, same metrics
(cents AVG/RMS error, FPR, weighted FNR, ROC/AUC) — so the numbers are directly comparable.

This is **benchmark/evaluation only**. Nothing here touches production code or mobile builds.
(Context: PESTO is LGPL-3.0, which rules out shipping its code/checkpoints in the app, but
internal benchmarking involves no distribution, so the license imposes nothing here. If PESTO
wins, deployment would later use self-trained weights + our own inference — out of scope now.)

## What PESTO is

- Repo: https://github.com/SonyCSLParis/pesto (inference; LGPL-3.0). Training repo: `pesto-full`.
- Tiny self-supervised pitch-estimation CNN (~29k params), monophonic, designed for real-time.
- Its README has a "NEW: Export Compiled Model" section: `python -m realtime.export_onnx --help`
  exports a **stateless streaming ONNX model** with an explicit cache tensor. Quoted inference
  cost ~0.7 ms/frame. Sample rate and step size are **frozen at export time** (encoded in the
  filename, e.g. `mir-1k_g7_44100_1024.onnx` = 44100 Hz, step 1024).

Reference usage from their README (Python, translates 1:1 to C++):

```python
session = ort.InferenceSession("mir-1k_g7_44100_1024.onnx")
cache_size = session.get_inputs()[1].shape[1]
cache_state = np.zeros((1, cache_size), dtype=np.float32)
for audio_chunk in audio_chunks:
    outputs = session.run(None, {"audio": audio_chunk, "cache": cache_state})
    prediction, confidence, volume, activations, cache_out = outputs
    cache_state = cache_out
```

## Step 1 — Export the ONNX model (one-time, Python)

1. Clone https://github.com/SonyCSLParis/pesto, `pip install` it (use a venv).
2. Run `python -m realtime.export_onnx --help` and inspect the options. **Important:** the
   benchmark feeds blocks of `sampleRate / 100` samples (10 ms — see
   `TestCaseUtils.h:159`). If the export supports a custom step size, export with
   **step = 10 ms at the corpus sample rate** (check the corpus WAVs' actual rate(s) first —
   `clean->sampleRate` comes from the test WAV files). Matching step size avoids FIFO
   re-chunking in the wrapper. If only fixed steps (e.g. 1024) are supported, export that and
   re-chunk in the wrapper (see below).
3. Verify the exported model's actual input/output names and shapes by introspection
   (`session.get_inputs()` / `get_outputs()`), don't trust the README ordering blindly.
4. **Verify `prediction` units** by checking the pesto source / running it on a known sine
   tone: it is most likely semitones (MIDI number), not Hz. Conversion:
   `hz = 440 * 2^((midi - 69) / 12)`. Confirm the reference pitch convention.
5. Put the `.onnx` file somewhere the test can find it (e.g. `saint/PitchDetector/Test/models/`,
   path passed via the existing `getArgument` mechanism or a CMake-configured path).

## Step 2 — C++ wrapper implementing `saint::PitchDetector`

Interface (`saint/PitchDetector/PitchDetector.h`):

```cpp
// returns 0.f if no pitch, else Hz. input = samplesPerBlock * numChannels, interleaved.
virtual float process(const float* input, DebugOutput* = nullptr,
                      std::vector<float>* debugOutputSignal = nullptr) = 0;
virtual int delaySamples() const = 0;
```

New class, suggested location `saint/PitchDetector/Test/PestoPitchDetector.{h,cpp}`
(test-only — do not add it to the production library target):

- Members: `Ort::Session`, cache tensor as `std::vector<float>` (re-fed each call),
  optional FIFO for re-chunking, the configured threshold.
- Use a single process-wide `Ort::Env` (static, e.g. Meyers singleton) shared by all
  instances; one `Ort::Session` per detector instance is simplest and matches the
  benchmark's threading model (it creates **one detector per test case** and runs test
  cases on multiple threads — see the manual threading in `PitchDetectorImplTests.cpp`).
- **Stereo:** the corpus may be stereo (interleaved). PESTO is mono — average the channels.
- **Re-chunking** (only if step ≠ blockSize): push each 10 ms block into a FIFO; every time
  ≥ step samples are buffered, run inference and store the result; `process()` returns the
  most recent result. Account for this buffering in `delaySamples()`.
- **`delaySamples()`:** at minimum the FIFO latency; the model's own analysis window adds
  more. The benchmark uses it to align estimates with ground-truth note on/offset times
  (`PitchDetectorImplTests.cpp` around line 141), so a wrong value skews FPR/FNR but not
  cents error. Start with step size (or window size if discoverable) and note it as a knob.
- **Voicing / outputs mapping:**
  - `confidence` → write into `DebugOutput` under key `"presenceScore"` (the ROC/AUC
    machinery reads exactly this key, `PitchDetectorImplTests.cpp:166`).
  - Return `0.f` when `confidence < threshold` (make the threshold configurable; start
    around 0.5 and let the ROC output suggest a better one).
  - `prediction` → convert to Hz (see Step 1.4) and return.
  - Ignore `volume` and `activations` for now.
- `debugOutputSignal`: can be left unimplemented (append nothing).

## Step 3 — Build

- Add ONNX Runtime to `saint/PitchDetector/Test/CMakeLists.txt` **only** (test target).
  Easiest: FetchContent/download of the prebuilt onnxruntime release for the host platform,
  or `find_package(onnxruntime)` if installed. CPU execution provider only; no GPU needed.
- Keep it optional if convenient (e.g. `SAINT_WITH_PESTO_BENCHMARK` CMake option) so the
  regular test build doesn't gain a hard dependency.

## Step 4 — Hook into the benchmark

The cleanest route: **refactor the body of `TEST(PitchDetectorImpl, benchmarking)`** so the
detector construction (currently the `Preprocessor`/`FrequencyDomainTransformer`/
`AutocorrPitchDetector`/... block, lines ~92-119) is supplied by a factory
`std::function<std::unique_ptr<PitchDetector>(const TestCase&)>`, then add a second test
`TEST(PestoPitchDetector, benchmarking)` using a factory that builds the PESTO wrapper.

Notes:

- The existing test ends with regression assertions against stored baselines
  (`previousRmsError = 7.09`, `previousAuc = 0.871`, `previousFNR = 0.403`, lines ~327-368).
  Those baselines describe SAINT — **do not apply them to the PESTO run**. For PESTO, just
  print/log the aggregate numbers (and write the same CSV/roc_curve.py outputs, ideally to a
  separate out dir so runs don't clobber each other).
- The existing `getArgument` mechanism (`indexOfProcessToLog`, `testCaseId`, ...) should keep
  working for the PESTO test too — it's how single-case debugging is done.
- The median filter / smoother wrapping (`testWithMedianFilter`) is SAINT post-processing;
  for a first pass run PESTO **raw** (its own temporal handling is internal). A follow-up
  could wrap PESTO in the same `PitchDetectorMedianFilter` + `PitchDetectionSmoother` for an
  apples-to-apples "as shipped" comparison — make it possible but not required.

## Success criteria / deliverable

1. `TEST(PestoPitchDetector, benchmarking)` runs over the full corpus and prints:
   cents AVG + RMS error, FPR, weighted FNR, AUC — same definitions as the SAINT test.
2. A short comparison table PESTO vs. the SAINT baselines (RMS ≈ 7.09 cents,
   AUC ≈ 0.871, FNR ≈ 0.403) plus observations: where PESTO wins/loses (low-E octave
   cases, high-noise mixes), and how well-calibrated its confidence is (ROC shape).
3. Note the effective latency implied by the wrapper (step size + buffering) vs. SAINT.

## Likely pitfalls

- ONNX input shapes: the `audio` input is probably `(1, step)` — check, and slice/copy
  accordingly (the benchmark hands you interleaved frames, not a batch dim).
- Sample-rate mismatch: if corpus WAVs are not all the same rate, either export one model
  per rate or resample in the wrapper. Check first; don't assume 44.1 kHz.
- `prediction` may be 0 / NaN / a sentinel when unvoiced — handle before converting to Hz.
- Don't let one shared cache leak across test cases: cache state is per-detector-instance,
  and a fresh instance per test case (as the harness already does) resets it naturally.
