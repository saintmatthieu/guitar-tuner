# Analysis of the autocorrelation peak census produced by AutocorrPeakCensusTests.
#
# Reads out/peakCensus_{cases,blocks,peaks}.csv and prints:
#   A. Census of peak classes relative to the ground-truth lag T (occurrence rates,
#      where the unconstrained max lands, recall of the true peak).
#   B. Value structure (relational features: v(T/2)/v(T), v(2T)/v(T), rank and margin
#      of the true peak).
#   C. Comb vs non-comb peak values, harmonic-noise check in pre-onset blocks.
# All main tables are computed on the headline stratum tOnset in [0.1, 1.0] s and
# conditioned on noise level and register.

import argparse
import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Ratio classes lag/T. r(T/2)=P_even-P_odd-type peaks come as half-integer combs, etc.
CLASSES = {
    "T/4": 0.25,
    "T/3": 1 / 3,
    "T/2": 0.5,
    "2T/3": 2 / 3,
    "3T/4": 0.75,
    "T": 1.0,
    "4T/3": 4 / 3,
    "3T/2": 1.5,
    "5T/3": 5 / 3,
    "2T": 2.0,
    "5T/2": 2.5,
    "3T": 3.0,
    "7T/2": 3.5,
    "4T": 4.0,
    "5T": 5.0,
    "6T": 6.0,
    "7T": 7.0,
    "8T": 8.0,
}
# Default chosen from the measured deviation of true T peaks (see
# peakCensusTolerance.py): q0.5%..q99.5% spans [-26.7, +17.8] cents in the low
# register (inharmonicity + per-note tuning offsets, systematically flat), so
# +/-25 covers ~99% of true peaks while halving the noise-credit window vs 50.
DEFAULT_TOLERANCE_CENTS = 25.0


def loadData(toleranceCents):
    cases = pd.read_csv(os.path.join(OUT_DIR, "peakCensus_cases.csv"))
    blocks = pd.read_csv(os.path.join(OUT_DIR, "peakCensus_blocks.csv"))
    peaks = pd.read_csv(
        os.path.join(OUT_DIR, "peakCensus_peaks.csv"),
        dtype={
            "caseIdx": np.int32,
            "blockIdx": np.int32,
            "lag": np.float32,
            "raw": np.float32,
            "corr": np.float32,
            "preNeg": np.int8,
        },
    )

    cases["trueLag"] = cases.sampleRate / cases.trueFreq
    cases["duration"] = cases.truthEnd - cases.truthStart
    cases["lastSearchIndex"] = np.minimum(
        cases.fftSize // 2, (cases.sampleRate / 69.2957).astype(int) # 69.2957 must be made configurable.
        # This experiment may have to be reproduced later with different search ranges (for other instruments, such as bass or ukulele)
    )
    # Noise level: noise RMS in dB; clean note is normalised to -10 dB peak, so
    # "-40" is the noisiest condition and "-inf" is clean.
    cases["noise"] = cases.snrDb.astype(str)
    bins = [0, 110, 220, 1000]
    cases["register"] = pd.cut(
        cases.trueFreq, bins, labels=["low (<110Hz)", "mid (110-220Hz)", "high (>220Hz)"]
    )

    caseCols = ["caseIdx", "noteFile", "trueFreq", "trueLag", "duration", "noise",
                "register", "lastSearchIndex"]
    blocks = blocks.merge(cases[caseCols], on="caseIdx")
    peaks = peaks.merge(cases[["caseIdx", "trueLag"]], on="caseIdx")

    # Classify each peak by its lag ratio to the true lag.
    ratio = peaks.lag.to_numpy() / peaks.trueLag.to_numpy()
    classNames = np.array(list(CLASSES.keys()))
    classRatios = np.array(list(CLASSES.values()))
    centsToClasses = np.abs(
        1200 * np.log2(ratio[:, None] / classRatios[None, :])
    )
    best = centsToClasses.argmin(axis=1)
    bestCents = centsToClasses[np.arange(len(best)), best]
    peakClass = np.where(bestCents <= toleranceCents, classNames[best], "other")
    peaks["cls"] = peakClass
    peaks["centsOff"] = 1200 * np.log2(ratio / classRatios[best])

    return cases, blocks, peaks


def addStratum(blocks):
    t = blocks.tOnset
    stratum = pd.Series("ignore", index=blocks.index)
    stratum[(t >= -0.5) & (t < -0.05)] = "pre-onset"
    stratum[(t >= 0.0) & (t < 0.1)] = "attack [0,0.1s)"
    stratum[(t >= 0.1) & (t <= 1.0) & (t <= blocks.duration)] = "head [0.1,1s]"
    stratum[(t > 1.0) & (t <= blocks.duration)] = "tail (1s,end]"
    blocks["stratum"] = stratum
    return blocks


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_CENTS,
                        help="class-assignment tolerance in cents")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(f"Class tolerance: +/-{args.tolerance:g} cents")
    print("Loading...", file=sys.stderr)
    cases, blocks, peaks = loadData(args.tolerance)
    blocks = addStratum(blocks)

    blockKey = ["caseIdx", "blockIdx"]

    # Per-block summary built from the peaks table -----------------------------------
    # The detector's unconstrained selection compares RAW values among post-crossing
    # peaks; the presence score uses the window-CORRECTED value.
    post = peaks[peaks.preNeg == 0]

    print("Computing per-block aggregates...", file=sys.stderr)
    idxRaw = post.groupby(blockKey, sort=False)["raw"].idxmax()
    maxRaw = post.loc[idxRaw, blockKey + ["cls", "raw", "corr"]].set_index(blockKey)
    maxRaw.columns = ["maxRawCls", "maxRawVal", "maxRawCorrVal"]
    idxCorr = post.groupby(blockKey, sort=False)["corr"].idxmax()
    maxCorr = post.loc[idxCorr, blockKey + ["cls", "corr"]].set_index(blockKey)
    maxCorr.columns = ["maxCorrCls", "maxCorrVal"]

    # Value of the comb positions per block (corrected), NaN if that peak is absent.
    combVals = {}
    for cls in ["T/2", "T", "2T", "3T", "T/3", "3T/2"]:
        sub = post[post.cls == cls].groupby(blockKey)["corr"].max()
        combVals[cls] = sub
    comb = pd.DataFrame(combVals)
    comb.columns = ["v_" + c for c in comb.columns]

    # Rank of the T peak among post-crossing peaks by corrected value.
    print("Computing ranks...", file=sys.stderr)
    post = post.copy()
    post["rankCorr"] = post.groupby(blockKey)["corr"].rank(ascending=False, method="min")
    post["rankRaw"] = post.groupby(blockKey)["raw"].rank(ascending=False, method="min")
    tPeaks = post[post.cls == "T"].groupby(blockKey).agg(
        tRankCorr=("rankCorr", "min"), tRankRaw=("rankRaw", "min")
    )
    # Whether the T peak exists at all, anywhere (incl. pre-crossing).
    tExists = peaks[peaks.cls == "T"].groupby(blockKey).size().rename("tExists") > 0
    tPreNegOnly = (
        peaks[peaks.cls == "T"].groupby(blockKey)["preNeg"].min().rename("tPreNegOnly") == 1
    )

    b = blocks.set_index(blockKey)
    b = b.join(maxRaw).join(maxCorr).join(comb).join(tPeaks).join(tExists).join(tPreNegOnly)
    b.tExists = b.tExists.fillna(False)
    b.tPreNegOnly = b.tPreNegOnly.fillna(False)

    head = b[b.stratum == "head [0.1,1s]"]
    # Blocks with enough signal that the question is meaningful: the detector itself
    # found *some* peak. Keep all head blocks; noise conditioning shows the rest.

    line = "=" * 100
    print(line)
    print("A2. Where does the unconstrained max land? (headline stratum [0.1s, 1s], % of blocks)")
    print(line)
    tab = (
        head.groupby(["noise"], observed=True)["maxRawCls"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    cols = [c for c in ["T", "T/2", "T/3", "3T/2", "2T", "3T", "other"] if c in tab.columns]
    otherCombs = tab.drop(columns=cols, errors="ignore").sum(axis=1)
    tab = (100 * tab[cols]).round(2)
    tab["otherComb"] = (100 * otherCombs).round(2)
    print(tab.to_string())
    print()
    print("Same, by register (all noise levels pooled):")
    tabReg = (
        head.groupby(["register"], observed=True)["maxRawCls"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    cols = [c for c in ["T", "T/2", "T/3", "3T/2", "2T", "3T", "other"] if c in tabReg.columns]
    print((100 * tabReg[cols]).round(2).to_string())
    print()
    print("Same, ranking by window-CORRECTED value instead of raw:")
    tabCorr = (
        head.groupby(["noise"], observed=True)["maxCorrCls"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    cols = [c for c in ["T", "T/2", "T/3", "3T/2", "2T", "3T", "other"] if c in tabCorr.columns]
    print((100 * tabCorr[cols]).round(2).to_string())

    print()
    print(line)
    print("A1. Peak-class occurrence: % of head-stratum blocks having a local max at each class")
    print(line)
    nHead = len(head)
    headPeaks = peaks.merge(
        head.reset_index()[blockKey + ["noise"]], on=blockKey, how="inner"
    )
    occ = (
        headPeaks.drop_duplicates(blockKey + ["cls"])
        .groupby("noise")["cls"]
        .value_counts()
        .unstack(fill_value=0)
        .div(head.groupby("noise").size(), axis=0)
    )
    cols = [c for c in ["T/3", "T/2", "T", "3T/2", "2T", "5T/2", "3T", "4T", "other"]
            if c in occ.columns]
    print((100 * occ[cols]).round(1).to_string())

    print()
    print(line)
    print("A3. Ceiling: is the true peak (class T) present as a local max?")
    print(line)
    for noise, gb in head.groupby("noise"):
        errBlocks = gb[gb.maxRawCls != "T"]
        print(
            f"noise {noise:>5}: T-peak present in {pct(gb.tExists.mean())} of blocks | "
            f"max!=T in {pct(len(errBlocks) / max(len(gb), 1))} of blocks, of which "
            f"T-peak still present: {pct(errBlocks.tExists.mean())} "
            f"(of those, hidden before neg-crossing: {pct(errBlocks[errBlocks.tExists].tPreNegOnly.mean()) if errBlocks.tExists.any() else 'n/a'})"
        )

    print()
    print(line)
    print("B5. Rank of the T peak (window-corrected value), head stratum, when max(raw) != T")
    print(line)
    err = head[(head.maxRawCls != "T") & head.tExists]
    rankCounts = err.tRankCorr.value_counts(normalize=True).sort_index()
    print("rank:  " + "  ".join(f"{int(r)}:{pct(p)}" for r, p in rankCounts.head(6).items()))

    print()
    print(line)
    print("B4. Relational separability (head stratum, blocks where both peaks exist)")
    print(line)
    both = head.dropna(subset=["v_T/2", "v_T"])
    both = both[both.maxRawCls.isin(["T", "T/2"])]
    r = both["v_T/2"] / both["v_T"]
    for name, gb in both.groupby(both.maxRawCls == "T/2"):
        label = "max=T/2 (octave-up err)" if name else "max=T   (correct)     "
        q = (r[gb.index]).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
        print(f"v(T/2)/v(T) | {label}: n={len(gb):7d}  q05={q.iloc[0]} q25={q.iloc[1]} "
              f"med={q.iloc[2]} q75={q.iloc[3]} q95={q.iloc[4]}")
    both2 = head.dropna(subset=["v_2T", "v_T"])
    both2 = both2[both2.maxRawCls.isin(["T", "2T"])]
    r2 = both2["v_2T"] / both2["v_T"]
    for name, gb in both2.groupby(both2.maxRawCls == "2T"):
        label = "max=2T (octave-down err)" if name else "max=T  (correct)       "
        q = (r2[gb.index]).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
        print(f"v(2T)/v(T)  | {label}: n={len(gb):7d}  q05={q.iloc[0]} q25={q.iloc[1]} "
              f"med={q.iloc[2]} q75={q.iloc[3]} q95={q.iloc[4]}")

    print()
    print(line)
    print("B6. Margin of error blocks: (vmax - vT)/vmax, corrected values, max(raw) != T")
    print(line)
    errWithT = err.dropna(subset=["v_T"])
    margin = (errWithT.maxRawCorrVal - errWithT["v_T"]) / errWithT.maxRawCorrVal
    q = margin.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
    print(f"n={len(errWithT)}  q05={q.iloc[0]} q25={q.iloc[1]} med={q.iloc[2]} "
          f"q75={q.iloc[3]} q95={q.iloc[4]}  (negative: T peak actually scores higher)")
    print(f"T peak scores higher (corrected) than the raw argmax in {pct((margin < 0).mean())} "
          "of error blocks")

    print()
    print(line)
    print("C7. Peak values: comb classes vs non-comb ('other'), head stratum, corrected value")
    print(line)
    headPost = headPeaks[headPeaks.preNeg == 0]
    isComb = headPost.cls != "other"
    for name, gb in headPost.groupby(isComb):
        label = "comb " if name else "other"
        q = gb["corr"].quantile([0.25, 0.5, 0.75, 0.95]).round(3)
        print(f"{label}: n={len(gb):8d}  q25={q.iloc[0]} med={q.iloc[1]} q75={q.iloc[2]} "
              f"q95={q.iloc[3]}")

    print()
    print(line)
    print("C9. Pre-onset blocks (noise only): strongest corrected peak value")
    print(line)
    pre = b[b.stratum == "pre-onset"]
    for noise, gb in pre.groupby("noise"):
        v = gb.maxCorrVal.dropna()
        if len(v) == 0:
            continue
        print(f"noise {noise:>5}: n={len(v):7d}  med={v.median():.3f}  q95={v.quantile(0.95):.3f}  "
              f">0.5: {pct((v > 0.5).mean())}  >0.8: {pct((v > 0.8).mean())}")

    print()
    print(line)
    print("D. Max-raw class by time since onset (all noise levels pooled)")
    print(line)
    tabT = (
        b[b.stratum.isin(["attack [0,0.1s)", "head [0.1,1s]", "tail (1s,end]"])]
        .groupby("stratum", observed=True)["maxRawCls"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    cols = [c for c in ["T", "T/2", "T/3", "3T/2", "2T", "3T", "other"] if c in tabT.columns]
    print((100 * tabT[cols]).round(2).to_string())

    print()
    print(line)
    print("Per-note aggregates (head stratum): notes where the max ever lands off T")
    print(line)
    noteAgg = head.groupby(["noteFile", "noise"], observed=True).agg(
        anyErr=("maxRawCls", lambda s: (s != "T").any()),
        errRate=("maxRawCls", lambda s: (s != "T").mean()),
    )
    byNoise = noteAgg.groupby("noise").agg(
        notesWithAnyErr=("anyErr", "mean"), avgErrRate=("errRate", "mean")
    )
    print((100 * byNoise).round(1).to_string())

    # Save the per-block table for further modeling.
    outPath = os.path.join(OUT_DIR, "peakCensus_blockSummary.parquet")
    try:
        b.reset_index().to_parquet(outPath)
        print(f"\nPer-block summary saved to {outPath}")
    except Exception as e:
        outPath = os.path.join(OUT_DIR, "peakCensus_blockSummary.csv.gz")
        b.reset_index().to_csv(outPath, index=False)
        print(f"\nPer-block summary saved to {outPath}")


if __name__ == "__main__":
    main()
