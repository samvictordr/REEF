#!/usr/bin/env python3
"""
Generate fig_ablation_saliency.pdf — omic saliency horizontal bar chart.

Mean gradient importance per omic layer under equal-weights conditions,
averaged across all LOOCV folds and 5 seeds.  Error bars show SD across
the 5 per-seed mean_importance_by_omic values (not within-fold SD).

Reads data from:
  outputs/ablation/equal_weights_results.json

Output:
  outputs/figures/fig_ablation_saliency.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_JSON = PROJECT_ROOT / "outputs" / "ablation" / "equal_weights_results.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "fig_ablation_saliency.pdf"

# Also copy to paper_figures for existing \includegraphics reference
PAPER_FIG_DIR = PROJECT_ROOT / "paper_figures"
PAPER_FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG_COPY = PAPER_FIG_DIR / "fig_ablation_saliency.pdf"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(ABLATION_JSON) as f:
    abl = json.load(f)

SEEDS = [42, 123, 456, 789, 1337]
OMICS_ORDER = ["proteomics", "microbiome", "metabolomics", "transcriptomics"]
OMIC_LABELS = ["Proteomics", "Microbiome", "Metabolomics", "Transcriptomics"]

# Collect per-seed importance for each omic
importance = {o: [] for o in OMICS_ORDER}
for s in SEEDS:
    imp = abl["per_seed_saliency_rankings"][str(s)]["mean_importance_by_omic"]
    for o in OMICS_ORDER:
        importance[o].append(imp[o])

means = [np.mean(importance[o]) for o in OMICS_ORDER]
sds = [np.std(importance[o], ddof=1) for o in OMICS_ORDER]

# ---------------------------------------------------------------------------
# Print verification
# ---------------------------------------------------------------------------
print("=== Omic saliency (mean ± SD across 5 seeds) ===")
for label, m, s in zip(OMIC_LABELS, means, sds):
    print(f"  {label:16s}: {m:.6f} ± {s:.6f}")
fold_ratio = means[0] / means[-1]
print(f"  Proteomics / Transcriptomics ratio: {fold_ratio:.1f}×")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 3.5))

y_pos = np.arange(len(OMIC_LABELS))
bars = ax.barh(
    y_pos, means, xerr=sds,
    height=0.55,
    color="#2166AC",
    alpha=0.8,
    edgecolor="black",
    linewidth=0.6,
    capsize=3,
    error_kw={"elinewidth": 1.0, "capthick": 1.0, "color": "black"},
    zorder=3,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(OMIC_LABELS, fontsize=8)
ax.invert_yaxis()  # Proteomics at top
ax.set_xlabel("Mean gradient importance", fontsize=9)
ax.set_xlim(0, 0.11)
ax.tick_params(axis="x", labelsize=8)

# Annotate each bar with its mean value
for i, (m, s) in enumerate(zip(means, sds)):
    x_text = m + s + 0.002
    if x_text > 0.105:
        # Place inside bar if it would overflow
        x_text = m - 0.003
        ha = "right"
        color = "white"
    else:
        ha = "left"
        color = "black"
    ax.text(x_text, i, f"{m:.3f}", va="center", ha=ha,
            fontsize=8, color=color, fontweight="medium")

# "~20×" annotation — vertical double-headed arrow between proteomics
# and transcriptomics bar ends
x_arrow = 0.098
y_top = 0  # proteomics index
y_bot = 3  # transcriptomics index
ax.annotate(
    "", xy=(x_arrow, y_top + 0.25), xytext=(x_arrow, y_bot - 0.25),
    arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.2),
)
ax.text(x_arrow + 0.003, (y_top + y_bot) / 2, "~20×",
        fontsize=8, color="#666666", va="center", ha="left",
        fontstyle="italic")

# Remove gridlines and unnecessary spines
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
fig.savefig(PAPER_FIG_COPY, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Copied: {PAPER_FIG_COPY}")
