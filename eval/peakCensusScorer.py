# Prototype of the multi-candidate Bayesian octave scorer discussed in
# peakCensusFindings.md (section C) and the follow-up.
#
# It does NOT cheat with truth-anchored values. For each head-stratum block it:
#   1. finds the max-raw post-crossing peak (the lag the current detector returns),
#      L_max;
#   2. builds octave candidates at L_max * {1/3, 1/2, 1, 2, 3}, kept only if their
#      lag is in the tuner's detectable range and an actual peak supports them;
#   3. for each candidate c (lag L_c) reads its comb by nearest-peak lookup and
#      forms the two relational features
#          rHalf = v(L_c/2) / v(L_c)   (octave-up evidence)
#          rDouble = v(2 L_c) / v(L_c) (octave-down evidence)
#      using the window-corrected peak values;
#   4. scores each candidate with a naive-Bayes generative model
#          score(c) = log f_H(rHalf | C) + log f_D(rDouble | C) + log prior(factor)
#      where f_H, f_D are the "candidate-is-fundamental" densities and the prior is
#      the octave-class distribution (the A2 census);
#   5. picks argmax as the predicted fundamental.
#
# Densities, priors, and accuracy are fit/evaluated with a per-recording
# (noteFile) train/test split so within-note correlation cannot leak.
#
# Reports per-block octave accuracy of the scorer vs the raw-max baseline, with the
# fixed/broken breakdown (introducing errors is the real risk).

import hashlib
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

MIN_FREQ_HZ = 69.2957   # Db2, lowest detectable (must be made configurable later)
MAX_FREQ_HZ = 369.994   # Gb4, highest detectable
READ_TOL_CENTS = 40.0   # nearest-peak acceptance window when reading a comb position
TRUE_TOL_CENTS = 25.0   # a candidate counts as the true fundamental within this
FLOOR = 0.02            # value assigned to an absent comb position
FACTORS = np.array([1 / 3, 1 / 2, 1.0, 2.0, 3.0])


class LogDensity1D:
    """Histogram density with Laplace smoothing, returning log pdf; values are
    clipped into [lo, hi] so the tails fold into the edge bins."""

    def __init__(self, values, lo, hi, bins=50, alpha=1.0):
        self.lo, self.hi, self.bins = lo, hi, bins
        counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
        width = (hi - lo) / bins
        dens = (counts + alpha) / ((len(values) + alpha * bins) * width)
        self.logdens = np.log(dens)
        self.edges = edges
        self.width = width

    def __call__(self, x):
        idx = np.clip(((np.clip(x, self.lo, self.hi) - self.lo) / self.width).astype(int),
                      0, self.bins - 1)
        return self.logdens[idx]


def fold_of(noteFile):
    return int(hashlib.md5(noteFile.encode()).hexdigest(), 16) % 2


def loadHeadPeaks():
    cases = pd.read_csv(os.path.join(OUT_DIR, "peakCensus_cases.csv"))
    cases["trueLag"] = cases.sampleRate / cases.trueFreq
    cases["duration"] = cases.truthEnd - cases.truthStart
    cases["noise"] = cases.snrDb.astype(str)
    cases["maxSearchLag"] = np.minimum(cases.fftSize // 2,
                                       (cases.sampleRate / MIN_FREQ_HZ).astype(int))
    cases["minSearchLag"] = cases.sampleRate / MAX_FREQ_HZ

    blocks = pd.read_csv(os.path.join(OUT_DIR, "peakCensus_blocks.csv"))
    blocks = blocks.merge(
        cases[["caseIdx", "noteFile", "trueLag", "duration", "noise",
               "maxSearchLag", "minSearchLag", "sampleRate"]],
        on="caseIdx")
    head = blocks[(blocks.tOnset >= 0.1) & (blocks.tOnset <= 1.0)
                  & (blocks.tOnset <= blocks.duration)].copy()
    headKeys = head[["caseIdx", "blockIdx"]]

    peaks = pd.read_csv(
        os.path.join(OUT_DIR, "peakCensus_peaks.csv"),
        dtype={"caseIdx": np.int32, "blockIdx": np.int32, "lag": np.float32,
               "raw": np.float32, "corr": np.float32, "preNeg": np.int8})
    peaks = peaks[peaks.preNeg == 0]
    peaks = peaks.merge(headKeys, on=["caseIdx", "blockIdx"], how="inner")
    peaks = peaks.sort_values(["caseIdx", "blockIdx", "lag"])
    return cases, head, peaks


def buildCandidates(head, peaks):
    """One row per (block, candidate); reads comb features by nearest-peak lookup."""
    info = {(r.caseIdx, r.blockIdx): (r.noteFile, r.noise, r.trueLag,
                                      r.minSearchLag, r.maxSearchLag)
            for r in head.itertuples()}
    rows = []
    tol = 2 ** (READ_TOL_CENTS / 1200) - 1  # fractional lag tolerance ~ cents

    for (ci, bi), g in peaks.groupby(["caseIdx", "blockIdx"], sort=False):
        lags = g.lag.to_numpy()
        vals = g["corr"].to_numpy()
        if len(lags) == 0:
            continue
        noteFile, noise, trueLag, minSearch, maxSearch = info[(ci, bi)]
        maxLag = float(lags[np.argmax(g.raw.to_numpy())])
        trueLag = float(trueLag)
        minSearch, maxSearch = float(minSearch), float(maxSearch)

        def read(target):
            if target <= 0:
                return FLOOR
            j = np.searchsorted(lags, target)
            best, bestErr = FLOOR, tol
            for k in (j - 1, j):
                if 0 <= k < len(lags):
                    err = abs(lags[k] - target) / target
                    if err <= bestErr:
                        best, bestErr = vals[k], err
            return best

        trueFactorOk = (abs(1200 * np.log2((maxLag * FACTORS) / trueLag))
                        <= TRUE_TOL_CENTS)
        for fi, f in enumerate(FACTORS):
            Lc = maxLag * f
            if Lc < minSearch or Lc > maxSearch:
                continue
            vSelf = read(Lc)
            if vSelf <= FLOOR:        # no real peak supports this candidate
                continue
            rows.append((ci, bi, noteFile, noise, f,
                         read(Lc / 2) / vSelf, read(2 * Lc) / vSelf,
                         bool(trueFactorOk[fi]), vSelf))
    cand = pd.DataFrame(rows, columns=["caseIdx", "blockIdx", "noteFile", "noise",
                                       "factor", "rHalf", "rDouble", "isTrue", "vSelf"])
    cand["fold"] = cand.noteFile.map(fold_of)
    return cand


def getCandidates():
    cachePath = os.path.join(OUT_DIR, "peakCensusScorer_cand.csv.gz")
    if os.path.exists(cachePath):
        print("Loading cached candidates...")
        return pd.read_csv(cachePath)
    print("Loading peaks (head stratum)...")
    cases, head, peaks = loadHeadPeaks()
    print(f"head blocks: {len(head)},  peaks: {len(peaks)}")
    print("Building candidates (anchored on the max peak)...")
    cand = buildCandidates(head, peaks)
    cand.to_csv(cachePath, index=False)
    return cand


def fitPrior(trainPos):
    blocks = trainPos.drop_duplicates(["caseIdx", "blockIdx"])
    counts = blocks.factor.value_counts()
    alpha = 1.0
    return {f: np.log((counts.get(f, 0) + alpha)
                      / (len(blocks) + alpha * len(FACTORS))) for f in FACTORS}


def scoreAndReport(name, test, scoreFn):
    test = test.copy()
    test["score"] = scoreFn(test)
    idx = test.groupby(["caseIdx", "blockIdx"])["score"].idxmax()
    winners = test.loc[idx, ["caseIdx", "blockIdx", "factor", "isTrue"]].rename(
        columns={"factor": "predFactor", "isTrue": "scorerRight"})
    blockTruth = (test.groupby(["caseIdx", "blockIdx"])
                  .agg(noise=("noise", "first"),
                       trueFactorPresent=("isTrue", "any"),
                       baselineRight=("isTrue", lambda s: bool(
                           s[test.loc[s.index, "factor"] == 1.0].any())))
                  .reset_index())
    m = blockTruth.merge(winners, on=["caseIdx", "blockIdx"], how="left")
    m["scorerRight"] = m.scorerRight.fillna(False)

    print(f"\n=== {name} ===")
    print(f"{'noise':>6} | {'n':>7} | {'baseline':>9} | {'scorer':>8} | "
          f"{'fixed':>6} | {'broken':>6} | {'net':>6}")
    for noise, gb in list(m.groupby("noise")) + [("ALL", m)]:
        base = gb.baselineRight.mean()
        scor = gb.scorerRight.mean()
        fixed = int(((~gb.baselineRight) & gb.scorerRight).sum())
        broken = int((gb.baselineRight & (~gb.scorerRight)).sum())
        print(f"{str(noise):>6} | {len(gb):>7} | {base:>8.2%} | {scor:>7.2%} | "
              f"{fixed:>6} | {broken:>6} | {fixed - broken:>+6}")
    return m, test


def main():
    cand = getCandidates()
    print(f"candidate rows: {len(cand)}")
    train, test = cand[cand.fold == 0], cand[cand.fold == 1]
    pos = train[train.isTrue]
    logPrior = fitPrior(pos)
    print("Priors (train, pooled): "
          + "  ".join(f"f={f:g}:{np.exp(logPrior[f]):.4f}" for f in FACTORS))

    # Model 1: pooled generative MAP (one fundamental density for all candidates).
    fH = LogDensity1D(pos.rHalf, 0.0, 1.5)
    fD = LogDensity1D(pos.rDouble, 0.0, 3.0)
    pooled = lambda t: (fH(t.rHalf.to_numpy()) + fD(t.rDouble.to_numpy())
                        + t.factor.map(logPrior).to_numpy())

    # Model 2: per-octave-class discriminative LLR. Fit positive and negative
    # densities separately FOR EACH factor (the signature of "I'm the fundamental"
    # differs by octave context), then score by log-likelihood ratio + prior.
    dens = {}
    for f in FACTORS:
        p, n = train[(train.factor == f) & train.isTrue], train[(train.factor == f) & ~train.isTrue]
        dens[f] = (LogDensity1D(p.rHalf, 0.0, 1.5), LogDensity1D(p.rDouble, 0.0, 3.0),
                   LogDensity1D(n.rHalf, 0.0, 1.5), LogDensity1D(n.rDouble, 0.0, 3.0))

    def perFactorLLR(t):
        s = np.full(len(t), -1e9)
        fac = t.factor.to_numpy()
        rh, rd = t.rHalf.to_numpy(), t.rDouble.to_numpy()
        for f in FACTORS:
            hP, dP, hN, dN = dens[f]
            mask = fac == f
            s[mask] = (hP(rh[mask]) - hN(rh[mask]) + dP(rd[mask]) - dN(rd[mask])
                       + logPrior[f])
        return s

    print("\n" + "=" * 84)
    print("Per-block octave accuracy on HELD-OUT recordings (test fold)")
    print("baseline = always keep the max peak (factor 1)")
    print("=" * 84)
    scoreAndReport("Model 1: pooled generative MAP", test, pooled)
    m2, _ = scoreAndReport("Model 2: per-octave-class discriminative LLR", test, perFactorLLR)

    print("\nModel 2 recovery of baseline errors, by true factor (test):")
    truth = test[test.isTrue].drop_duplicates(["caseIdx", "blockIdx"])[
        ["caseIdx", "blockIdx", "factor"]]
    err = m2[~m2.baselineRight & m2.trueFactorPresent].merge(
        truth, on=["caseIdx", "blockIdx"], how="left")
    print(err.groupby("factor")["scorerRight"].agg(["size", "mean"])
          .rename(columns={"size": "n_baseline_errors", "mean": "scorer_recovers"})
          .to_string())

    # Decision-margin sweep: override the max only when the best alternative beats
    # the factor-1 candidate's score by tau. tau=0 is the pure argmax; tau->inf is
    # the baseline. This is the precision/recall (fixed-vs-broken) operating knob.
    t = test.copy()
    t["score"] = perFactorLLR(t)
    isMax = t.factor == 1.0
    maxC = t[isMax].set_index(["caseIdx", "blockIdx"])
    others = t[~isMax]
    oidx = others.groupby(["caseIdx", "blockIdx"])["score"].idxmax()
    bestOther = others.loc[oidx].set_index(["caseIdx", "blockIdx"])
    j = maxC[["score", "isTrue"]].rename(columns={"score": "sMax", "isTrue": "maxRight"}).join(
        bestOther[["score", "isTrue"]].rename(columns={"score": "sOther", "isTrue": "otherRight"}))
    j["sOther"] = j.sOther.fillna(-1e18)
    j["otherRight"] = j.otherRight.fillna(False)
    baseAcc = j.maxRight.mean()
    print("\nDecision-margin sweep (Model 2 LLR), test fold, all noise pooled:")
    print(f"  baseline (keep max)         acc={baseAcc:.4%}")
    for tau in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]:
        override = (j.sOther - j.sMax) > tau
        right = np.where(override, j.otherRight, j.maxRight)
        fixed = int((override & ~j.maxRight & j.otherRight).sum())
        broken = int((override & j.maxRight & ~j.otherRight).sum())
        print(f"  tau={tau:>4.1f}  acc={right.mean():.4%}  fixed={fixed:>4}  "
              f"broken={broken:>4}  net={fixed - broken:>+5}")


if __name__ == "__main__":
    main()
