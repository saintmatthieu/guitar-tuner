# How far do TRUE comb peaks deviate from their nominal lag positions?
# Informs the choice of tolerances in peakCensus.py and in a future algorithm.
#
# Two separate questions, two sections:
#
# 1. TRUTH-RELATIVE (analysis labeling): for each head-stratum block in clean /
#    low-noise conditions, find the ACF peak nearest to each comb position (T, T/2,
#    2T) within a generous +/-100 cents of the GROUND TRUTH, and look at the
#    deviation distribution — decomposed into the per-note median offset
#    (instrument tuning + inharmonicity, systematic) and the within-note residual
#    (block-to-block ACF noise).
#
# 2. PEAK-RELATIVE (what the algorithm sees): anchor on the refined T peak itself
#    and measure where the comb partners sit relative to ratio * anchorLag. The
#    per-note tuning offset cancels here, so this distribution — not the
#    truth-relative one — dictates how wide the algorithm's pairwise search window
#    (around 2L, L/2, 3L of a found peak at L) must be.

import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    cases = pd.read_csv(os.path.join(OUT_DIR, "peakCensus_cases.csv"))
    blocks = pd.read_csv(os.path.join(OUT_DIR, "peakCensus_blocks.csv"))
    peaks = pd.read_csv(os.path.join(OUT_DIR, "peakCensus_peaks.csv"))

    cases["trueLag"] = cases.sampleRate / cases.trueFreq
    cases["duration"] = cases.truthEnd - cases.truthStart
    cases["noise"] = cases.snrDb.astype(str)
    bins = [0, 110, 220, 1000]
    cases["register"] = pd.cut(
        cases.trueFreq, bins, labels=["low (<110Hz)", "mid (110-220Hz)", "high (>220Hz)"]
    )

    blocks = blocks.merge(
        cases[["caseIdx", "noteFile", "noise", "register", "trueLag", "duration"]], on="caseIdx"
    )
    head = blocks[
        (blocks.tOnset >= 0.1) & (blocks.tOnset <= 1.0) & (blocks.tOnset <= blocks.duration)
    ]
    # Clean and least-noisy conditions only: deviations here are signal-driven
    # (tuning, inharmonicity, refinement bias), not noise-driven.
    head = head[head.noise.isin(["-inf", "-60", "-60.0"])]

    peaks = peaks.merge(head[["caseIdx", "blockIdx", "noteFile", "register", "trueLag"]],
                        on=["caseIdx", "blockIdx"], how="inner")

    print(f"head-stratum clean/-60 blocks: {head.groupby(['caseIdx','blockIdx']).ngroups}")

    def q(s):
        qq = s.quantile([0.005, 0.05, 0.5, 0.95, 0.995]).round(1)
        return (f"q0.5%={qq.iloc[0]:6.1f}  q5%={qq.iloc[1]:6.1f}  med={qq.iloc[2]:6.1f}  "
                f"q95%={qq.iloc[3]:6.1f}  q99.5%={qq.iloc[4]:6.1f}")

    print("\n--- 1. Deviation relative to the GROUND TRUTH (analysis labeling) ---")
    for clsName, ratio in [("T", 1.0), ("T/2", 0.5), ("2T", 2.0)]:
        cents = 1200 * np.log2(peaks.lag / (peaks.trueLag * ratio))
        near = peaks[np.abs(cents) <= 100].copy()
        near["centsOff"] = cents[near.index]
        # Nearest peak to the class position per block.
        idx = near.groupby(["caseIdx", "blockIdx"])["centsOff"].apply(
            lambda s: s.abs().idxmin()
        )
        sel = near.loc[idx]

        noteMedian = sel.groupby("noteFile")["centsOff"].transform("median")
        residual = sel.centsOff - noteMedian

        print(f"\n=== class {clsName}  (n={len(sel)}) ===")
        print(f"total deviation     : {q(sel.centsOff)}")
        print(f"per-note offsets    : {q(sel.groupby('noteFile')['centsOff'].median())}")
        print(f"within-note residual: {q(residual)}")
        print("total deviation by register:")
        for reg, gb in sel.groupby("register", observed=True):
            print(f"  {reg:16s}: {q(gb.centsOff)}  (n={len(gb)})")

    print("\n--- 2. Deviation relative to the ANCHOR PEAK (the algorithm's view) ---")
    print("anchor = refined peak nearest the truth (within +/-50c); partner = nearest")
    print("peak to ratio*anchorLag within +/-100c\n")
    cents = 1200 * np.log2(peaks.lag / peaks.trueLag)
    nearT = peaks[np.abs(cents) <= 50].copy()
    nearT["absCents"] = np.abs(cents[nearT.index])
    anchorIdx = nearT.groupby(["caseIdx", "blockIdx"])["absCents"].idxmin()
    anchors = nearT.loc[anchorIdx, ["caseIdx", "blockIdx", "lag"]].rename(
        columns={"lag": "anchorLag"}
    )
    withAnchor = peaks.merge(anchors, on=["caseIdx", "blockIdx"])

    for partnerName, ratio in [("T/2", 0.5), ("3T/2", 1.5), ("2T", 2.0), ("3T", 3.0)]:
        relCents = 1200 * np.log2(withAnchor.lag / (withAnchor.anchorLag * ratio))
        nearPartner = withAnchor[np.abs(relCents) <= 100].copy()
        # Exclude the anchor itself (only an issue for ratio 1, not listed).
        nearPartner["relCents"] = relCents[nearPartner.index]
        idx = nearPartner.groupby(["caseIdx", "blockIdx"])["relCents"].apply(
            lambda s: s.abs().idxmin()
        )
        sel = nearPartner.loc[idx]
        print(f"partner {partnerName:4s} vs anchor (n={len(sel)}): {q(sel.relCents)}")
        for reg, gb in sel.groupby("register", observed=True):
            print(f"  {reg:16s}: {q(gb.relCents)}  (n={len(gb)})")


if __name__ == "__main__":
    main()
