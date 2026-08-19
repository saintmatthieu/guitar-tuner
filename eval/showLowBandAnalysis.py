"""Shows what the below-range ("too low") verdict was based on, for one analysis frame.

The below-range decision (LowBandAnalyzer) asks whether a confident in-range estimate is
merely a harmonic of a string sounding below the tuning's range. It answers that from its own
long-window, further-decimated spectrum: it fits a harmonic comb whose fundamental sits below
the range, and measures how much of that comb the in-range estimate cannot account for. This
plots exactly that, so the answer can be checked by eye.

Produce the log first - one test case, one frame:

    PitchDetectorImplTests "algorithm=impl" \\
        "testCaseId=testFiles/notes/Martin DX1/A1_1.wav | testFiles/noise/AC_noise.wav | -50" \\
        "indexOfProcessToLog=457"

Pick the frame from eval/out/frameDump.csv, written by the same run (its `frame` column is the
index indexOfProcessToLog takes): a row where lowBandSupport is high shows the verdict being
made, one where it is low shows why it was not.

    python3 eval/showLowBandAnalysis.py [out.png]

With a file name, writes the figure instead of opening a window.
"""

import csv
import os
import sys

import matplotlib

if len(sys.argv) > 1:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

outDir = os.path.join(os.path.dirname(__file__), "out")
sys.path.insert(0, outDir)

import PitchDetectorLog as pdl  # noqa: E402

if not hasattr(pdl, "lowBandSpectrum"):
    sys.exit("This log has no below-range analysis in it. Re-run the benchmark with "
             "indexOfProcessToLog=<frame>.")

# ---- the analysed spectrum -------------------------------------------------------------
hzPerBin = pdl.lowBandRate / pdl.lowBandFftSize
spectrum = pdl.lowBandSpectrum[: pdl.lowBandFftSize // 2]
f = [i * hzPerBin for i in range(len(spectrum))]

combHz = pdl.lowBandCombHz
combProminence = pdl.lowBandCombProminence
explained = [e > 0.5 for e in pdl.lowBandCombExplained]
topHz = max(combHz) * 1.1 if combHz else 700.0

verdictHz = pdl.lowBandVerdictHz
support = pdl.lowBandVerdictSupport
floor = pdl.lowBandSupportFloor
accepted = support >= floor

kEvidence = "#1b7f4b"  # comb members the in-range estimate does not explain: the evidence
kExplained = "#9aa0a6"  # comb members that are its own harmonics anyway: no evidence either way

frameDump = os.path.join(outDir, "frameDump.csv")
haveDump = os.path.exists(frameDump)

fig = plt.figure(figsize=(13, 9 if haveDump else 6.5))
rows = 3 if haveDump else 2
gs = fig.add_gridspec(rows, 2, height_ratios=[1.5, 1] + ([1.2] if haveDump else []))

# ---- 1: the spectrum the decision reads, with the fitted comb on it --------------------
ax = fig.add_subplot(gs[0, :])
ax.plot(f, spectrum, color="#3c4043", linewidth=0.9, zorder=2)
ax.axhline(0, color="#c0392b", linestyle="--", linewidth=1,
           label=f"noise floor ({pdl.lowBandFloorDb:.0f} dB)")
for hz, prom, isExplained in zip(combHz, combProminence, explained):
    ax.axvline(hz, color=kExplained if isExplained else kEvidence,
               linestyle="-" if not isExplained else ":", linewidth=1.4, alpha=0.75, zorder=1)
    if prom > 0.5:
        # Axes fraction for y: the data limits are still growing as artists are added.
        ax.annotate(f"{prom:.0f}", xy=(hz, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color=kExplained if isExplained else kEvidence)
if hasattr(pdl, "truthHz"):
    ax.axvline(pdl.truthHz, color="#1a73e8", linewidth=2.2, alpha=0.5,
               label=f"truth {pdl.truthHz:.1f} Hz")
if hasattr(pdl, "searchRangeHz"):
    ax.axvspan(pdl.searchRangeHz[0], topHz, color="#1a73e8", alpha=0.05)
    ax.axvline(pdl.searchRangeHz[0], color="#1a73e8", linestyle="--", linewidth=1,
               label=f"range floor {pdl.searchRangeHz[0]:.1f} Hz")
ax.axvline(pdl.lowBandInRangeHz, color="#e8710a", linewidth=1.6,
           label=f"in-range estimate {pdl.lowBandInRangeHz:.1f} Hz")
ax.set_xlim(0, topHz)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("dB above noise floor")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
frameLabel = f"frame {pdl.loggedFrame}" if hasattr(pdl, "loggedFrame") else "logged frame"
ax.set_title(
    f"Below-range analysis, {frameLabel}: fundamental {verdictHz:.1f} Hz, "
    f"support {support:.2f} vs floor {floor:.2f} → "
    + ("BELOW RANGE" if accepted else "not accepted")
)

# ---- 2: the comb, harmonic by harmonic ------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ms = list(range(1, len(combProminence) + 1))
ax.bar(ms, combProminence,
       color=[kExplained if e else kEvidence for e in explained])
ax.set_xticks(ms)
ax.set_xlabel(f"harmonic of {verdictHz:.1f} Hz")
ax.set_ylabel("prominence (dB)")
ax.grid(True, axis="y", alpha=0.3)
ax.set_title("Where the support comes from", fontsize=10)
ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=kEvidence),
    plt.Rectangle((0, 0), 1, 1, color=kExplained)],
    labels=["the in-range estimate cannot explain it", "it is the estimate's own harmonic"],
    fontsize=8, loc="upper right")

# ---- 3: why this candidate won --------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
candidates = pdl.lowBandCandidateHz
x = range(len(candidates))
bars = ax.bar(x, pdl.lowBandCandidateSupport,
              color=["#1b7f4b" if abs(c - verdictHz) < 1e-3 else "#a8c7b5" for c in candidates])
ax.axhline(floor, color="#c0392b", linestyle="--", linewidth=1, label=f"support floor {floor}")
for i, prom in enumerate(pdl.lowBandCandidateProminence):
    ax.annotate(f"{prom:.0f} dB total", (i, 0.03), ha="center", va="bottom", fontsize=7,
                rotation=90, color="#3c4043")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{c:.1f}" for c in candidates])
ax.set_xlabel("candidate fundamental (Hz) - the winner has the most total prominence")
ax.set_ylabel("support")
ax.set_ylim(0, 1)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Candidates considered", fontsize=10)

# ---- 4: the whole case, so one frame is not mistaken for the behaviour ----------------
if haveDump:
    with open(frameDump) as fh:
        rowsIn = list(csv.DictReader(fh))
    frames = [int(r["frame"]) for r in rowsIn]
    supports = [float(r["lowBandSupport"]) for r in rowsIn]
    below = [float(r["bucket"]) == 0 for r in rowsIn]
    ax = fig.add_subplot(gs[2, :])
    ax.plot(frames, supports, color=kEvidence, linewidth=0.8, label="support")
    ax.axhline(floor, color="#c0392b", linestyle="--", linewidth=1, label="support floor")
    ax.fill_between(frames, 0, 1, where=below, color="#1b7f4b", alpha=0.15,
                    label="reported below range")
    if hasattr(pdl, "loggedFrame"):
        ax.axvline(pdl.loggedFrame, color="#e8710a", linewidth=1.4, label="this frame")
    ax.set_xlim(min(frames), max(frames))
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame")
    ax.set_ylabel("support")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.set_title("The whole case (eval/out/frameDump.csv)", fontsize=10)

fig.suptitle("Below-range detection - what the verdict was based on")
fig.tight_layout()
fig.canvas.manager.set_window_title("Below-range analysis")

if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], dpi=110, bbox_inches="tight")
    print("wrote", sys.argv[1])
else:
    plt.show()
