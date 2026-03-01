#!/usr/bin/env python3
"""
Generate fig2_seed_variability.pdf — 4-condition stability figure.

Panel (A): Violin plots showing AUROC distribution across 5 seeds for
           REEF, Equal-weights ablation, LASER, NVFlare.
Panel (B): Per-seed trajectory (line+marker) for each method.

Reads data from:
  outputs/benchmark/comprehensive_benchmark.json
  outputs/ablation/equal_weights_results.json

Output:
  outputs/figures/fig2_seed_variability.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('pdf')  # vector PDF backend, no GUI
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_JSON = PROJECT_ROOT / "outputs" / "benchmark" / "comprehensive_benchmark.json"
ABLATION_JSON = PROJECT_ROOT / "outputs" / "ablation" / "equal_weights_results.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "fig2_seed_variability.pdf"

# Also copy to paper_figures for existing \includegraphics reference
PAPER_FIG_DIR = PROJECT_ROOT / "paper_figures"
PAPER_FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG_COPY = PAPER_FIG_DIR / "fig2_seed_variability.pdf"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(BENCHMARK_JSON) as f:
    bench = json.load(f)
with open(ABLATION_JSON) as f:
    abl = json.load(f)

SEEDS = [42, 123, 456, 789, 1337]

# Try new key first, fall back to old key for backward compatibility
_reef_key = "reef" if "reef" in bench["methods"] else "domain_aware_vfl"
da_aurocs = bench["methods"][_reef_key]["aurocs"]
laser_aurocs = bench["methods"]["laser"]["aurocs"]
nvflare_aurocs = bench["methods"]["nvflare_vfl"]["aurocs"]
ablation_aurocs = [abl["per_seed_auroc"][str(s)] for s in SEEDS]

# ---------------------------------------------------------------------------
# Print verification
# ---------------------------------------------------------------------------
print("=== Per-seed AUROCs used for fig2 ===")
for name, arr in [("REEF", da_aurocs), ("Ablation", ablation_aurocs),
                  ("LASER", laser_aurocs), ("NVFlare", nvflare_aurocs)]:
    print(f"  {name:12s}: {[round(v, 4) for v in arr]}  "
          f"mean={np.mean(arr):.4f}  SD={np.std(arr, ddof=0):.4f}")

# ---------------------------------------------------------------------------
# Colours (colorblind-safe, greyscale-distinguishable)
# ---------------------------------------------------------------------------
COLORS = {
    "REEF":    "#2166AC",  # dark blue
    "Ablation": "#F4A582",  # salmon/orange
    "LASER":    "#92C5DE",  # light blue
    "NVFlare":  "#BABABA",  # grey
}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(7.5, 5.5),
    gridspec_kw={"width_ratios": [55, 45], "wspace": 0.30},
)

# --- Panel labels ---
for ax, label in [(ax_a, "(A)"), (ax_b, "(B)")]:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="right")

# ---------------------------------------------------------------------------
# Panel A — Violin plots
# ---------------------------------------------------------------------------
labels_a = ["REEF", "Ablation\n(equal-wt)", "LASER", "NVFlare"]
data_a = [da_aurocs, ablation_aurocs, laser_aurocs, nvflare_aurocs]
colors_a = [COLORS["REEF"], COLORS["Ablation"],
            COLORS["LASER"], COLORS["NVFlare"]]

positions_a = range(len(labels_a))
parts = ax_a.violinplot(data_a, positions=positions_a, showmeans=False,
                        showmedians=False, showextrema=False)

for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(colors_a[i])
    body.set_edgecolor("black")
    body.set_linewidth(0.8)
    body.set_alpha(0.55)

# Overlay individual seed points with distinct markers per condition
markers_a = ["o", "s", "^", "D"]
for i, (arr, marker) in enumerate(zip(data_a, markers_a)):
    jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(arr))
    ax_a.scatter(
        [i + j for j in jitter], arr,
        color=colors_a[i], edgecolors="black", linewidths=0.6,
        s=36, marker=marker, zorder=5,
    )

# Mean markers (horizontal bar)
for i, arr in enumerate(data_a):
    mean_val = np.mean(arr)
    ax_a.hlines(mean_val, i - 0.22, i + 0.22,
                colors="black", linewidths=1.5, zorder=6)

# Chance line
ax_a.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
ax_a.text(len(labels_a) - 0.5, 0.505, "Chance", fontsize=7, color="#888888",
          ha="right", va="bottom")

ax_a.set_xticks(range(len(labels_a)))
ax_a.set_xticklabels(labels_a, fontsize=8)
ax_a.set_ylabel("AUROC", fontsize=9)
ax_a.set_ylim(0.2, 1.05)
ax_a.tick_params(axis="y", labelsize=8)
ax_a.set_xlim(-0.6, len(labels_a) - 0.4)

# Remove gridlines & top/right spines
ax_a.grid(False)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# Panel B — Per-seed trajectory
# ---------------------------------------------------------------------------
methods_b = [
    ("REEF",    da_aurocs,      "#2166AC", "-",  "o"),
    ("Ablation\n(equal-wt)", ablation_aurocs, "#F4A582", "--", "s"),
    ("LASER",    laser_aurocs,   "#92C5DE", ":",  "^"),
    ("NVFlare",  nvflare_aurocs, "#BABABA", ":",  "D"),
]

x_b = range(len(SEEDS))
for label, arr, color, ls, marker in methods_b:
    ax_b.plot(x_b, arr, color=color, linestyle=ls, marker=marker,
              markersize=6, markeredgecolor="black", markeredgewidth=0.5,
              linewidth=1.8, label=label, zorder=4)

# Chance line
ax_b.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, zorder=1)

ax_b.set_xticks(list(x_b))
ax_b.set_xticklabels([str(s) for s in SEEDS], fontsize=8)
ax_b.set_xlabel("Random seed", fontsize=9)
ax_b.set_ylabel("AUROC", fontsize=9)
ax_b.set_ylim(0.2, 1.05)
ax_b.tick_params(axis="y", labelsize=8)

ax_b.legend(loc="upper right", fontsize=7, framealpha=0.9,
            edgecolor="#cccccc", handlelength=2.2)

ax_b.grid(False)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
fig.savefig(PAPER_FIG_COPY, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Copied: {PAPER_FIG_COPY}")
