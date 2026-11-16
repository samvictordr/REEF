#!/usr/bin/env python3
"""Paper figures + summary.json for the v4 benchmark manuscript."""
import json
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, nogueira, boot_ci  # noqa: E402

OUT = "figs"
os.makedirs(OUT, exist_ok=True)

# reference palette (light mode), fixed order by job
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEL_COLOR = {"none": "#52514e", "saliency": "#2a78d6", "random": "#eb6834", "t-test": "#1baf7a"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})

ARMS = [  # (key, backbone, selector, budget, label)
    ("vfl_full", "VFL", "none", "-", "VFL, all features"),
    ("reef", "VFL", "saliency", "expert", "VFL, saliency, expert + weights (v1–3)"),
    ("reef_equal", "VFL", "saliency", "equal", "VFL, saliency, equal"),
    ("reef_random", "VFL", "random", "expert", "VFL, random, expert + weights"),
    ("equal_random", "VFL", "random", "equal", "VFL, random, equal"),
    ("equal_ttest", "VFL", "t-test", "equal", "VFL, t-test, equal"),
    ("laser_full", "LASER", "none", "-", "LASER, all features"),
    ("laser_reefsel", "LASER", "saliency", "expert", "LASER, saliency, expert"),
    ("laser_salQ", "LASER", "saliency", "equal", "LASER, saliency, equal"),
    ("laser_rndE", "LASER", "random", "expert", "LASER, random, expert"),
    ("laser_rndQ", "LASER", "random", "equal", "LASER, random, equal"),
    ("laser_ttQ", "LASER", "t-test", "equal", "LASER, t-test, equal"),
    ("enet", "Central", "none", "-", "Elastic net (centralised)"),
]
LABEL = {a[0]: a[4] for a in ARMS}


def loocv_auc(path):
    rows, _ = load(path)
    lab = {r["fold"]: r["labels"][0] for r in rows}
    n = len(lab)
    y = np.array([lab[i] for i in range(n)])
    by = defaultdict(dict)
    for r in rows:
        for a, v in r["arms"].items():
            if "p" in v:
                by[(a, r["seed"])][r["fold"]] = v["p"][0]
    out = defaultdict(list)
    for (a, s), d in sorted(by.items()):
        if len(d) == n:
            out[a].append(roc_auc_score(y, [d[i] for i in range(n)]))
    return {a: np.array(v) for a, v in out.items()}, rows


def rskf_matrix(path):
    rows, _ = load(path)
    fa = defaultdict(list)
    for r in rows:
        for a, v in r["arms"].items():
            if "p" in v:
                fa[(a, r["seed"], r["split"])].append(roc_auc_score(r["labels"], v["p"]))
    seeds = sorted({r["seed"] for r in rows})
    splits = sorted({r["split"] for r in rows})
    M = {}
    for a, *_ in ARMS:
        m = np.full((len(seeds), len(splits)), np.nan)
        for i, s in enumerate(seeds):
            for j, sp in enumerate(splits):
                v = fa.get((a, s, sp))
                if v and len(v) == 3:
                    m[i, j] = np.mean(v)
        if not np.isnan(m).all():
            M[a] = m
    return M, rows


summary = {}

# ---------------------------------------------------------------- data
loocv, loocv_rows = loocv_auc("loocv.jsonl+loocv_laser.jsonl")
rsk_path = "rskf3.jsonl+rskf3_laser.jsonl" if os.path.exists("rskf3_laser.jsonl") else "rskf3.jsonl"
rsk, rsk_rows = rskf_matrix(rsk_path)

summary["loocv"] = {a: {"mean": float(v.mean()), "ci": [float(x) for x in boot_ci(v)], "sd": float(v.std(ddof=1)),
                        "min": float(v.min()), "max": float(v.max()), "n": int(len(v))} for a, v in loocv.items()}
summary["rskf3"] = {}
for a, M in rsk.items():
    split_means = np.nanmean(M, 0)
    seed_sd = np.nanstd(M, 0, ddof=1)
    summary["rskf3"][a] = {"mean": float(np.nanmean(M)), "ci": [float(x) for x in boot_ci(split_means)],
                           "seed_sd": float(np.nanmean(seed_sd)) if a != "enet" else 0.0,
                           "split_sd": float(np.nanstd(split_means, ddof=1))}

# original-code reproduction
orig = defaultdict(list)
for f in sorted(glob.glob("../runs/orig/seed*/comprehensive_benchmark.json")):
    r = json.load(open(f))
    for m, v in r["methods"].items():
        orig[m].append(v["statistics"]["auroc"]["mean"])
summary["orig"] = {m: {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for m, v in orig.items()}


# ---------------------------------------------------------------- Fig 1: reproduction vs corrected
def strip(ax, x, vals, color, filled=True):
    jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.18
    ax.scatter(x + jit, vals, s=16, facecolor=color if filled else "white", edgecolor=color, linewidth=1, zorder=3)
    ax.plot([x - 0.17, x + 0.17], [np.mean(vals)] * 2, color=INK, lw=1.6, zorder=4, solid_capstyle="round")


fig, ax = plt.subplots(figsize=(5.2, 2.6))
pairs = [("nvflare_vfl", "vfl_full", "VFL, all features"), ("reef", "reef", "v1–3 saliency pipeline"), ("laser", "laser_full", "LASER-VFL, all features")]
for i, (o, c, name) in enumerate(pairs):
    strip(ax, i * 1.0 - 0.2, np.array(orig[o]), "#8d8c87", filled=True)
    strip(ax, i * 1.0 + 0.2, loocv[c], "#2a78d6", filled=True)
ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.text(-0.45, 0.505, "chance", color=INK2, fontsize=7, va="bottom", ha="left")
ax.set_xticks(range(3))
ax.set_xticklabels([p[2] for p in pairs])
ax.set_ylabel("LOOCV AUROC")
ax.set_ylim(0.15, 1.04)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
h1 = plt.Line2D([], [], marker="o", ls="", color="#8d8c87", label="Versions 1–3 code (single-sample early stopping, 5 seeds)")
h2 = plt.Line2D([], [], marker="o", ls="", color="#2a78d6", label="Corrected protocol (fixed 50 steps, 20 seeds)")
ax.legend(handles=[h1, h2], loc="lower left", fontsize=7, bbox_to_anchor=(0, 1.0), ncol=1)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_reproduction.pdf")
plt.close(fig)


# ---------------------------------------------------------------- Fig 2: forest plot
def forest(ax, stats, title, key_mean="mean", rows=None):
    order = rows or [a for a, *_ in ARMS if a in stats]
    ys = []
    y = 0
    prev = None
    for a in order:
        bb = next(x[1] for x in ARMS if x[0] == a)
        if prev is not None and bb != prev:
            y += 0.6
        ys.append(y)
        prev = bb
        y += 1
    for a, yy in zip(order, ys):
        if a not in stats:
            continue
        _, bb, sel, bud, _ = next(x for x in ARMS if x[0] == a)
        s = stats[a]
        c = SEL_COLOR[sel]
        lo, hi = s["ci"]
        ax.plot([lo, hi], [yy, yy], color=c, lw=1.6, solid_capstyle="round", zorder=2)
        mk = "D" if bb == "Central" else "o"
        face = "white" if bud == "equal" else c
        ax.scatter([s[key_mean]], [yy], marker=mk, s=26, facecolor=face, edgecolor=c, linewidth=1.3, zorder=3)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(ys)
    ax.set_ylim(max(ys) + 0.7, -0.7)
    ax.set_xlim(0.35, 1.03)
    ax.set_xlabel("AUROC (mean, 95% CI)")
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return order, ys


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
ROWS = [a for a, *_ in ARMS if a in summary["rskf3"] or a in summary["loocv"]]
order, ys = forest(axes[0], summary["rskf3"], "A  Repeated stratified 3-fold (primary)", rows=ROWS)
axes[0].set_yticklabels([LABEL[a] for a in order])
forest(axes[1], summary["loocv"], "B  Pooled LOOCV", rows=ROWS)
hs = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in SEL_COLOR.items()]
hs += [plt.Line2D([], [], marker="o", ls="", markerfacecolor=INK2, markeredgecolor=INK2, label="expert budget"),
       plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, label="equal budget")]
fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.6, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{OUT}/fig2_benchmark.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: seed vs split variance
fig, ax = plt.subplots(figsize=(5.2, 3.4))
order = [a for a, *_ in ARMS if a in summary["rskf3"] and a != "enet"]
for i, a in enumerate(order):
    s = summary["rskf3"][a]
    ax.plot([s["seed_sd"], s["split_sd"]], [i, i], color=GRID, lw=1.2, zorder=1)
    ax.scatter([s["seed_sd"]], [i], s=22, color="#2a78d6", zorder=3)
    ax.scatter([s["split_sd"]], [i], s=22, color="#eb6834", zorder=3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([LABEL[a] for a in order])
ax.invert_yaxis()
ax.set_xlabel("Standard deviation of AUROC")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#2a78d6", label="across training seeds (within partition)"),
                   plt.Line2D([], [], marker="o", ls="", color="#eb6834", label="across data partitions (seed-averaged)")],
          loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_variance.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: permutation test
if os.path.exists("perm.jsonl"):
    prow, _ = load("perm.jsonl")
    # statistic: AUROC of the 13 pooled out-of-fold predictions of one 3-fold partition
    # (a permuted test fold can contain a single class, so per-fold AUROC is not always defined)
    null = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for r in prow:
        for a, v in r["arms"].items():
            if "p" in v:
                null[a][r["permute_seed"]][0].extend(r["labels"])
                null[a][r["permute_seed"]][1].extend(v["p"])
    obs = defaultdict(lambda: ([], []))
    for r in rsk_rows:
        if r["seed"] == 0 and r["split"] == 0:
            for a, v in r["arms"].items():
                if "p" in v:
                    obs[a][0].extend(r["labels"])
                    obs[a][1].extend(v["p"])
    perm_arms = [a for a in ["vfl_full", "reef", "equal_random", "laser_full", "laser_rndQ", "enet"] if a in null and a in obs]
    summary["perm"] = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True)
    for ax, a in zip(axes.flat, perm_arms):
        nv = np.array([roc_auc_score(yy, pp) for yy, pp in null[a].values() if len(yy) == 13])
        o = float(roc_auc_score(*obs[a]))
        p = (np.sum(nv >= o) + 1) / (len(nv) + 1)
        summary["perm"][a] = {"observed": o, "null_mean": float(nv.mean()), "null_sd": float(nv.std(ddof=1)), "p": float(p), "m": int(len(nv))}
        ax.hist(nv, bins=np.linspace(0, 1, 21), color="#c3c2b7", edgecolor="white", linewidth=0.8)
        ax.axvline(o, color="#2a78d6", lw=1.8)
        ax.set_title(f"{LABEL[a]}\nobserved {o:.2f}, p = {p:.3f}", fontsize=7, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel("Pooled out-of-fold AUROC")
    for ax in axes[:, 0]:
        ax.set_ylabel("Permutations")
    for ax in list(axes.flat)[len(perm_arms):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_permutation.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- Fig 5: sensitivity
sens = {}
for R in (20, 50, 100):
    f = "loocv.jsonl+loocv_laser.jsonl" if R == 50 else f"sens_r{R}.jsonl"
    if os.path.exists(f.split("+")[0]):
        auc, _ = loocv_auc(f)
        sens[R] = {a: v[:10] for a, v in auc.items()}  # first 10 seeds for comparability
lr_auc = None
if os.path.exists("sens_laserlr1e-4.jsonl"):
    lr_auc, _ = loocv_auc("sens_laserlr1e-4.jsonl")
    summary["sens_laser_lr1e-4"] = {a: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for a, v in lr_auc.items()}
if len(sens) == 3:
    summary["sens_rounds"] = {R: {a: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for a, v in d.items()} for R, d in sens.items()}
    sarms = ["vfl_full", "reef", "equal_random", "laser_full", "laser_rndQ"]
    cols = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    for k, (a, c) in enumerate(zip(sarms, cols)):
        xs = [i for i, R in enumerate((20, 50, 100)) if a in sens[R]]
        m = np.array([sens[(20, 50, 100)[i]][a].mean() for i in xs])
        sd = np.array([sens[(20, 50, 100)[i]][a].std(ddof=1) for i in xs])
        xo = np.array(xs) + (k - 2) * 0.09
        ax.vlines(xo, m - sd, m + sd, color=c, lw=1.2, alpha=0.7)
        ax.plot(xo, m, color=c, lw=2, marker="o", ms=5, label=LABEL[a])
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["20", "50", "100"])
    ax.set_xlabel("Optimiser steps")
    ax.set_ylabel("LOOCV AUROC (mean ± SD, 10 seeds)")
    ax.set_title("A  Training length", fontsize=8, loc="left")
    ax.set_ylim(0.3, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)
    if lr_auc is not None:
        for k, (a, c) in enumerate(zip(["laser_full", "laser_rndQ"], ["#1baf7a", "#6250d6"])):
            hi, lo = sens[50][a], lr_auc[a]
            xo = np.array([0, 1]) + (k - 0.5) * 0.12
            m = np.array([hi.mean(), lo.mean()])
            sd = np.array([hi.std(ddof=1), lo.std(ddof=1)])
            bx.vlines(xo, m - sd, m + sd, color=c, lw=1.2, alpha=0.7)
            bx.plot(xo, m, color=c, lw=2, marker="o", ms=5, label=LABEL[a])
        bx.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
        bx.set_xticks([0, 1])
        bx.set_xticklabels(["$10^{-3}$", "$10^{-4}$"])
        bx.set_xlim(-0.5, 1.5)
        bx.set_ylim(0.3, 1.02)
        bx.set_xlabel("LASER-VFL learning rate (50 steps)")
        bx.set_title("B  LASER-VFL learning rate", fontsize=8, loc="left")
        bx.grid(axis="y", color=GRID, lw=0.6)
        bx.set_axisbelow(True)
        bx.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_sensitivity.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- selection stability
p_of = {"transcriptomics": 62038, "proteomics": 4054, "metabolomics": 12055, "microbiome": 12432}
stab = {}
for a in ["reef", "reef_equal", "equal_ttest"]:
    masks = defaultdict(dict)
    for r in loocv_rows:
        v = r["arms"].get(a, {})
        if "mask" in v:
            masks[r["seed"]][r["fold"]] = v["mask"]
    stab[a] = {}
    for o in p_of:
        s_seed = np.mean([nogueira([masks[s][f][o] for s in masks], p_of[o]) for f in range(13)])
        s_fold = np.mean([nogueira([masks[s][f][o] for f in masks[s]], p_of[o]) for s in masks])
        stab[a][o] = {"across_seeds": float(s_seed), "across_folds": float(s_fold)}
summary["stability"] = stab

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps({k: list(v.keys()) if isinstance(v, dict) else v for k, v in summary.items()}, indent=0)[:2000])
