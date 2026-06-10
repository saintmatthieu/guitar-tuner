<!-- _class: lead -->

# 🎸 Benchmarking the Pitch Detector

#### How we measure accuracy — and catch regressions automatically

---

## Why benchmark?

- **Objective** — real numbers, not "sounds about right"
- **Safe** — a quality drop fails the build
- **Directed** — tune the algorithm from evidence, not hunches

---

<!-- _class: lead -->

> Run the full pipeline over **thousands of realistic recordings**,
> score **every 10 ms block**,
> and **fail CI** if quality drops.

---

## One big test matrix

```
        110 clean notes
              ×
     12 noise types  +  silence
              ×
      −40  /  −50  /  −60 dB  SNR
              =
       ~4,000 test cases
```

Every run exercises the *whole* space — instrument × noise × loudness.

---

## Built to mirror reality

- **Notes** — real instruments, single pluck, left to ring out
  ~16 instruments · acoustic & electric · mic'd & DI
- **Noise** — AC · PC fan · kitchen · TV · voices … recorded separately
- **Mixed by the harness** at calibrated SNR → fully reproducible
- Sample mix follows real usage
  *(5% of users play ukulele ⇒ ~5% ukulele samples)*

---

## Every 10 ms block is labelled

```
 │  before   │██████ note sustain ██████│   decay   │
 │ ── neg ── │ ───────  positive  ───── │ ── neg ── │
```

**Should a pitch be detected here — yes or no?**
That label is what every metric is scored against.

---

## Four metrics

| Metric | The question it answers |
|---|---|
| **RMS error** (cents) | When we report a pitch, how accurate is it? |
| **FNR** | How often do we *miss* a real note? |
| **FPR** | How often do we *invent* one? |
| **AUC** | How separable are note vs. no-note overall? |

---

<!-- _class: lead -->

## Where we stand today

### ≈ **7 cents**  · RMS error
### ≈ **28 %**  · miss rate (incl. extreme −40 dB noise)
### ≈ **0.87**  · AUC, at ≤ 1 % false alarms

*Human pitch JND ≈ 5–6 cents → most errors are sub-perceptual.*

---

## Accuracy, visualised

(Run benchmarking and show histogram.)

99.7 % of estimates land within ±50 cents — only tiny octave-error tails.

---

## Automated regression gate

- Each metric has a **stored reference value**
- Every run compared at **±1 % tolerance**
- 🔴 worse → test **fails** (CI red)
- 🟢 better → we bump the reference **on purpose**, with a reason

---

## When something regresses

- `benchmarking.csv` — per-case breakdown → the worst offenders
- `eval/show*.py` — ROC curve, error histograms, frequency tracks
- Pinpoint *which* instrument / noise / SNR broke

---

<!-- _class: lead -->

## Takeaways

- Large, realistic, **reproducible** test set
- Accuracy **and** detection quality — in numbers
- **CI-gated**: quality can't regress silently
- Rich diagnostics for when it does
