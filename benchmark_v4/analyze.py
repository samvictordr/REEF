#!/usr/bin/env python3
"""Summarise pilot outputs: means, seed vs split variance, paired tests, selection stability."""
import json
import sys
import itertools
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ARMS = ["reef", "reef_equal", "reef_random", "equal_random", "equal_ttest",
        "vfl_full", "laser_full", "laser_reefsel", "laser_salQ", "laser_rndE", "laser_rndQ", "laser_ttQ", "enet"]
rng = np.random.default_rng(0)


def load(path):
    """Read a jsonl; merge arms from any '+'-joined extra files keyed by task."""
    parts = path.split("+")
    rows = [json.loads(l) for l in open(parts[0])]
    idx = {(r["scheme"], r["seed"], r["split"], r["fold"]): r for r in rows}
    for extra in parts[1:]:
        for l in open(extra):
            e = json.loads(l)
            idx[(e["scheme"], e["seed"], e["split"], e["fold"])]["arms"].update(e["arms"])
    errs = sum(1 for r in rows for v in r["arms"].values() if "error" in v)
    return rows, errs


def boot_ci(x, f=np.mean, B=5000):
    x = np.asarray(x)
    bs = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def sd_ratio_ci(a, b, B=5000):
    a, b = np.asarray(a), np.asarray(b)
    bs = [np.std(a[rng.integers(0, len(a), len(a))], ddof=1) /
          max(np.std(b[rng.integers(0, len(b), len(b))], ddof=1), 1e-12) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def nogueira(masks, p):
    """Nogueira et al. 2018 stability for a list of selected-index sets over p features."""
    M = len(masks)
    Z = np.zeros((M, p))
    for i, m in enumerate(masks):
        Z[i, m] = 1
    k = Z.sum(1).mean()
    s2 = Z.var(0, ddof=1).mean()
    return 1 - s2 / ((k / p) * (1 - k / p))


def loocv(path):
    rows, errs = load(path)
    print(f"\n=== LOOCV ({len(rows)} tasks, {errs} arm errors) ===")
    by = defaultdict(dict)  # (arm, seed) -> {fold: p}
    labels = {}
    for r in rows:
        labels[r["fold"]] = r["labels"][0]
        for a, v in r["arms"].items():
            if "p" in v:
                by[(a, r["seed"])][r["fold"]] = v["p"][0]
    seeds = sorted({r["seed"] for r in rows})
    n = len(labels)
    y = np.array([labels[i] for i in range(n)])
    auc = {}
    probs = {}
    for a in ARMS:
        vals, P = [], []
        for s in seeds:
            d = by.get((a, s), {})
            if len(d) == n:
                p = np.array([d[i] for i in range(n)])
                vals.append(roc_auc_score(y, p))
                P.append(p)
        if vals:
            auc[a] = np.array(vals)
            probs[a] = np.array(P)
    print(f"{'arm':15s} {'n':>3s} {'mean':>6s} {'95%CI':>15s} {'SD':>6s} {'min':>6s} {'max':>6s} {'per-sample p SD':>16s} {'acc':>5s}")
    for a, v in auc.items():
        lo, hi = boot_ci(v)
        psd = probs[a].std(0, ddof=1).mean()
        acc = ((probs[a] >= 0.5) == y).mean()
        print(f"{a:15s} {len(v):3d} {v.mean():6.3f} [{lo:5.3f},{hi:5.3f}] {v.std(ddof=1):6.3f} {v.min():6.3f} {v.max():6.3f} {psd:16.3f} {acc:5.3f}")
    if "reef" in auc:
        print("\nREEF vs arm (LOOCV, seeds): mean diff [paired-by-seed 95% CI], SD ratio (arm/REEF) [boot CI], Brown-Forsythe p")
        for a, v in auc.items():
            if a == "reef":
                continue
            m = min(len(v), len(auc["reef"]))
            d = auc["reef"][:m] - v[:m]
            lo, hi = boot_ci(d)
            rlo, rhi = sd_ratio_ci(v, auc["reef"])
            bf = stats.levene(v, auc["reef"], center="median").pvalue if v.std() > 0 or auc["reef"].std() > 0 else float("nan")
            r = v.std(ddof=1) / max(auc["reef"].std(ddof=1), 1e-12)
            print(f"  {a:15s} diff {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  SDratio {r:5.2f} [{rlo:4.2f},{rhi:5.2f}]  BF p={bf:.3g}")
    # selection stability across seeds within fold, and across folds within seed
    for a in ["reef", "reef_equal", "equal_ttest"]:
        masks = defaultdict(dict)
        for r in rows:
            v = r["arms"].get(a, {})
            if "mask" in v:
                masks[r["seed"]][r["fold"]] = v["mask"]
        if not masks:
            continue
        omics = list(next(iter(next(iter(masks.values())).values())).keys())
        p_of = {"transcriptomics": 62038, "proteomics": 4054, "metabolomics": 12055, "microbiome": 12432}
        for o in omics:
            across_seeds = np.mean([nogueira([masks[s][f][o] for s in masks if f in masks[s]], p_of[o]) for f in range(n)])
            across_folds = np.mean([nogueira([masks[s][f][o] for f in masks[s]], p_of[o]) for s in masks])
            print(f"  selection stability {a:12s} {o:15s} across-seeds {across_seeds:6.3f}  across-folds {across_folds:6.3f}")
    return auc


def rskf(path):
    rows, errs = load(path)
    print(f"\n=== Repeated stratified 3-fold ({len(rows)} tasks, {errs} arm errors) ===")
    # per (arm, seed, split): concatenate the 3 test folds -> one AUROC over all 13 samples,
    # plus mean of per-fold AUROCs
    acc = defaultdict(lambda: {"p": [], "y": [], "fa": []})
    for r in rows:
        for a, v in r["arms"].items():
            if "p" in v:
                k = (a, r["seed"], r["split"])
                acc[k]["p"] += v["p"]
                acc[k]["y"] += r["labels"]
                acc[k]["fa"].append(roc_auc_score(r["labels"], v["p"]))
    seeds = sorted({r["seed"] for r in rows})
    splits = sorted({r["split"] for r in rows})
    A = {}
    for a in ARMS:
        M = np.full((len(seeds), len(splits)), np.nan)
        for i, s in enumerate(seeds):
            for j, sp in enumerate(splits):
                d = acc.get((a, s, sp))
                if d and len(d["fa"]) == 3:
                    M[i, j] = np.mean(d["fa"])
        if not np.isnan(M).all():
            A[a] = M
    print(f"{'arm':15s} {'mean':>6s} {'95%CI(splits)':>15s} {'seedSD':>7s} {'splitSD':>8s} {'totalSD':>8s}")
    for a, M in A.items():
        seed_sd = np.nanmean(np.nanstd(M, 0, ddof=1))      # across seeds, within split
        split_sd = np.nanstd(np.nanmean(M, 0), ddof=1)      # across splits, seed-averaged
        lo, hi = boot_ci(np.nanmean(M, 0))
        print(f"{a:15s} {np.nanmean(M):6.3f} [{lo:5.3f},{hi:5.3f}] {seed_sd:7.3f} {split_sd:8.3f} {np.nanstd(M, ddof=1):8.3f}")
    if "reef" in A:
        print("\nREEF vs arm: mean diff, Nadeau-Bengio corrected t-test over 30 split-folds (seed-averaged); seed-SD ratio [boot over splits]")
        # fold-level differences, seed averaged
        fold_auc = defaultdict(dict)
        for r in rows:
            for a, v in r["arms"].items():
                if "p" in v:
                    fold_auc[a].setdefault((r["split"], r["fold"]), []).append(roc_auc_score(r["labels"], v["p"]))
        keys = sorted(fold_auc["reef"])
        n_te, n_tr = 13 / 3, 13 * 2 / 3
        for a in A:
            if a == "reef":
                continue
            d = np.array([np.mean(fold_auc["reef"][k]) - np.mean(fold_auc[a][k]) for k in keys if k in fold_auc[a]])
            J = len(d)
            var = d.var(ddof=1) * (1 / J + n_te / n_tr)
            t = d.mean() / np.sqrt(var) if var > 0 else float("nan")
            p = 2 * stats.t.sf(abs(t), J - 1)
            sr = np.nanstd(A[a], 0, ddof=1) / np.maximum(np.nanstd(A["reef"], 0, ddof=1), 1e-12)
            lo, hi = boot_ci(sr, f=np.mean)
            print(f"  {a:15s} diff {d.mean():+.3f}  corrected p={p:.3g}   seed-SD ratio (arm/REEF) {np.mean(sr):5.2f} [{lo:4.2f},{hi:5.2f}]")
    return A


if __name__ == "__main__":
    for p in sys.argv[1:]:
        (loocv if "loocv" in p else rskf)(p)
