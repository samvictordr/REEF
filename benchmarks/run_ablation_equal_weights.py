#!/usr/bin/env python3
"""
Equal-Weights Ablation Study
Removes domain priors from REEF while holding total feature count
at 1,300. Tests whether biological priors specifically provide stability advantage
vs. any aggressive dimensionality reduction.

Changes from main benchmark (ONLY these two):
  1. omic_topk_ratios: all 0.65 (equal 325 features per omic, total 1300)
  2. omic_biological_weights: all 1.0 (no omic layer favored)

All other hyperparameters, seeds, LOOCV structure, scaler discipline,
and number of rounds are identical to the main domain-aware benchmark.
"""

import sys
import json
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from scipy import stats

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import NVFlare-native implementations (REAL NVFlare APIs)
from src.nvflare_vfl.nvflare_native_trainer import (
    DomainAwareNVFlareTrainer,
    DomainAwareNVFlareConfig,
    load_coral_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Ablation Configuration
# =============================================================================

ABLATION_CONFIG = {
    # --- ABLATION: these are the ONLY differences from the main domain-aware config ---
    "omic_topk_ratios": {
        "transcriptomics": 0.65,  # ABLATION: equal weights (was 0.3 in main benchmark)
        "proteomics":      0.65,  # ABLATION: equal weights (was 0.5 in main benchmark)
        "metabolomics":    0.65,  # ABLATION: equal weights (was 1.0 in main benchmark)
        "microbiome":      0.65,  # ABLATION: equal weights (was 0.8 in main benchmark)
    },
    "omic_biological_weights": {
        "transcriptomics": 1.0,   # ABLATION: equal weights (was 1.5 in main benchmark)
        "proteomics":      1.0,   # ABLATION: equal weights (was 1.0 in main benchmark)
        "metabolomics":    1.0,   # ABLATION: equal weights (was 0.8 in main benchmark)
        "microbiome":      1.0,   # ABLATION: equal weights (was 0.5 in main benchmark)
    },
    # --- Everything below is IDENTICAL to main domain-aware config ---
    "topk_features":       500,   # identical to main benchmark
    "n_rounds":            50,    # identical to main benchmark
    "hidden_dim":          128,   # identical to main benchmark
    "embedding_dim":       64,    # identical to main benchmark
    "lr_client":           5e-5,  # identical to main benchmark
    "lr_server":           5e-4,  # identical to main benchmark
    "weight_decay":        1e-4,  # identical to main benchmark
    "dropout":             0.3,   # identical to main benchmark
    "patience":            15,    # identical to main benchmark
    "batch_size":          8,     # identical to main benchmark
    "domain_warmup_rounds": 20,   # identical to main benchmark
}

SEEDS = [42, 123, 456, 789, 1337]


# =============================================================================
# Seed Function (copied exactly from main benchmark for self-containment)
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds."""
    random.seed(seed)  # FIX: seed Python random for LASER task sampling reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Metric Computation (copied from main benchmark for self-containment)
# =============================================================================

def compute_extended_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute extended classification metrics."""
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        precision_score, recall_score,
    )

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auroc"] = 0.5

    return metrics


# =============================================================================
# Ablation LOOCV Runner
# =============================================================================

def run_equal_weights_ablation_loocv(
    data_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    """
    Run REEF LOOCV with equal weights (no domain priors).

    Structure is identical to run_reef_loocv() in the main
    benchmark, except omic_topk_ratios and omic_biological_weights are
    overridden to equal values.

    Also captures per-fold gradient saliency importance for secondary analysis.
    """
    logger.info(f"  Running Equal-Weights Ablation LOOCV (seed={seed})...")
    set_seed(seed)

    start_time = datetime.now()

    data, labels, splits = load_coral_data(data_dir, seed=seed, cv_mode="loocv")
    n_samples = splits["n_samples"]

    all_preds = np.zeros(n_samples)
    all_labels = np.zeros(n_samples)
    fold_metrics = []
    total_features_selected = []

    # Saliency tracking per fold
    fold_importance_by_omic = []  # list of dicts, one per fold
    dominant_omic_by_fold = []

    for fold_idx in range(n_samples):
        test_idx = np.array([fold_idx])
        train_idx = np.array([i for i in range(n_samples) if i != fold_idx])

        if len(train_idx) > 2:
            val_idx = train_idx[:1]
            train_idx_actual = train_idx[1:]
        else:
            val_idx = train_idx[:1]
            train_idx_actual = train_idx

        config = DomainAwareNVFlareConfig(
            n_rounds=ABLATION_CONFIG["n_rounds"],            # identical to main benchmark
            topk_features=ABLATION_CONFIG["topk_features"],  # identical to main benchmark
            domain_warmup_rounds=ABLATION_CONFIG["domain_warmup_rounds"],  # identical to main benchmark
            batch_size=min(ABLATION_CONFIG["batch_size"], len(train_idx_actual)),  # identical to main benchmark
            lr_client=ABLATION_CONFIG["lr_client"],          # identical to main benchmark
            lr_server=ABLATION_CONFIG["lr_server"],          # identical to main benchmark
            weight_decay=ABLATION_CONFIG["weight_decay"],    # identical to main benchmark
            hidden_dim=ABLATION_CONFIG["hidden_dim"],        # identical to main benchmark
            embedding_dim=ABLATION_CONFIG["embedding_dim"],  # identical to main benchmark
            dropout=ABLATION_CONFIG["dropout"],              # identical to main benchmark
            patience=ABLATION_CONFIG["patience"],            # identical to main benchmark
            seed=seed + fold_idx,                            # identical to main benchmark
            # --- ABLATION overrides ---
            omic_topk_ratios=ABLATION_CONFIG["omic_topk_ratios"],
            omic_biological_weights=ABLATION_CONFIG["omic_biological_weights"],
        )

        trainer = DomainAwareNVFlareTrainer(data, labels, config)

        try:
            history = trainer.train(train_idx_actual, val_idx)
            test_metrics = trainer.evaluate(test_idx, return_predictions=True)

            preds = test_metrics.get("predictions", np.array([0.5]))
            all_preds[fold_idx] = float(preds[0]) if len(preds) > 0 else 0.5
            all_labels[fold_idx] = labels[fold_idx]

            total_selected = sum(
                s.get("selected_features", 0) for s in trainer.omic_stats.values()
            )
            total_features_selected.append(total_selected)

            # --- Capture gradient saliency importance (secondary observation) ---
            # trainer.domain_importance is set during Phase 2 (_apply_domain_selection)
            if hasattr(trainer, 'domain_importance') and trainer.domain_importance:
                fold_imp = {}
                for omic_name, imp_array in trainer.domain_importance.items():
                    fold_imp[omic_name] = float(np.mean(imp_array))
                fold_importance_by_omic.append(fold_imp)

                dominant = max(fold_imp, key=fold_imp.get)
                dominant_omic_by_fold.append(dominant)
            else:
                fold_importance_by_omic.append({})
                dominant_omic_by_fold.append("unknown")

            fold_metrics.append({
                "fold": fold_idx,
                "prediction": all_preds[fold_idx],
                "label": int(labels[fold_idx]),
                "features_selected": total_selected,
            })
        except Exception as e:
            logger.warning(f"    Fold {fold_idx} failed: {e}")
            all_preds[fold_idx] = 0.5
            all_labels[fold_idx] = labels[fold_idx]
            fold_importance_by_omic.append({})
            dominant_omic_by_fold.append("error")

    # Feature count verification
    if total_features_selected:
        avg_features = float(np.mean(total_features_selected))
        if abs(avg_features - 1300) > 1:
            logger.warning(
                f"  WARNING: Average features selected = {avg_features:.0f}, expected 1300!"
            )
        else:
            logger.info(f"  Feature count verified: {avg_features:.0f} (expected 1300)")

    overall_metrics = compute_extended_metrics(all_labels, all_preds)
    train_time = (datetime.now() - start_time).total_seconds()

    # Compute mean importance across folds
    omic_names = list(ABLATION_CONFIG["omic_topk_ratios"].keys())
    mean_importance_by_omic = {}
    for omic_name in omic_names:
        vals = [fi.get(omic_name, 0.0) for fi in fold_importance_by_omic if fi]
        mean_importance_by_omic[omic_name] = float(np.mean(vals)) if vals else 0.0

    return {
        "seed": seed,
        "test_auroc": overall_metrics["auroc"],
        "test_accuracy": overall_metrics["accuracy"],
        "test_f1": overall_metrics.get("f1", 0.0),
        "n_folds": n_samples,
        "train_time": train_time,
        "avg_features_selected": float(np.mean(total_features_selected)) if total_features_selected else 0,
        "fold_metrics": fold_metrics,
        "saliency": {
            "mean_importance_by_omic": mean_importance_by_omic,
            "dominant_omic_by_fold": dominant_omic_by_fold,
        },
    }


# =============================================================================
# Feature Allocation Verification
# =============================================================================

def verify_feature_allocations(data_dir: Path) -> Dict[str, int]:
    """
    Verify feature allocations at runtime without running full training.
    Returns the expected number of features per omic with equal ratios.
    """
    base_k = ABLATION_CONFIG["topk_features"]
    ratio = 0.65
    topk = int(base_k * ratio)  # = 325

    data, _, _ = load_coral_data(data_dir, seed=42, cv_mode="loocv")

    allocations = {}
    for omic_name, features in data.items():
        n_features = features.shape[1]
        actual_topk = min(topk, n_features, 2000)
        actual_topk = max(actual_topk, 50)
        allocations[omic_name] = actual_topk

    allocations["total"] = sum(v for k, v in allocations.items() if k != "total")
    return allocations


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("EQUAL-WEIGHTS ABLATION STUDY")
    logger.info("REEF with domain priors removed")
    logger.info("=" * 70)

    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "outputs" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Verify feature allocations ---
    logger.info("\nVerifying feature allocations...")
    allocations = verify_feature_allocations(data_dir)
    logger.info(f"  Feature allocations: {allocations}")

    if allocations["total"] != 1300:
        logger.error(
            f"  FATAL: Total features = {allocations['total']}, expected 1300. "
            "Ratio arithmetic is wrong. Aborting."
        )
        sys.exit(1)
    logger.info(f"  Total features verified: {allocations['total']} == 1300 ✓")

    # --- Step 2: Run ablation across all seeds ---
    per_seed_aurocs = {}
    per_seed_results = {}
    per_seed_saliency = {}

    for seed_idx, seed in enumerate(SEEDS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed} ({seed_idx + 1}/{len(SEEDS)})")
        logger.info(f"{'='*60}")

        result = run_equal_weights_ablation_loocv(data_dir, seed)

        per_seed_aurocs[str(seed)] = result["test_auroc"]
        per_seed_results[str(seed)] = result
        per_seed_saliency[str(seed)] = result["saliency"]

        logger.info(
            f"  Seed {seed} AUROC: {result['test_auroc']:.4f} "
            f"(features: {result['avg_features_selected']:.0f})"
        )

    # --- Step 3: Compute aggregate statistics ---
    auroc_values = np.array(list(per_seed_aurocs.values()))
    mean_auroc = float(auroc_values.mean())
    sd_auroc = float(auroc_values.std(ddof=0))  # population std
    cv = sd_auroc / mean_auroc if mean_auroc > 0 else float('inf')
    n = len(auroc_values)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci_margin = t_crit * sd_auroc / np.sqrt(n)
    ci_95 = [float(mean_auroc - ci_margin), float(mean_auroc + ci_margin)]

    # --- Step 4: Compute saliency dominance summary ---
    all_dominant = []
    for seed_str, sal in per_seed_saliency.items():
        all_dominant.extend(sal["dominant_omic_by_fold"])
    dominance_counts = dict(Counter(all_dominant))
    # Ensure all omics are represented
    for omic in ABLATION_CONFIG["omic_topk_ratios"].keys():
        if omic not in dominance_counts:
            dominance_counts[omic] = 0

    # --- Step 5: Load main benchmark results for comparison ---
    main_results_path = output_dir.parent / "benchmark" / "comprehensive_benchmark.json"
    comparison = {}
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_data = json.load(f)

        # Try new key first, fall back to old key for backward compatibility
        _reef_key = "reef" if "reef" in main_data["methods"] else "domain_aware_vfl"
        da_aurocs = main_data["methods"][_reef_key]["aurocs"]
        da_mean = float(np.mean(da_aurocs))
        da_sd = float(np.std(da_aurocs, ddof=0))

        # Paired t-test
        ablation_arr = np.array([per_seed_aurocs[str(s)] for s in SEEDS])
        da_arr = np.array(da_aurocs)

        t_stat, t_pvalue = stats.ttest_rel(da_arr, ablation_arr)

        # Cohen's d
        diffs = da_arr - ablation_arr
        cohens_d = float(diffs.mean() / (diffs.std(ddof=0) + 1e-10))

        # SD ratio
        sd_ratio = sd_auroc / da_sd if da_sd > 0 else float('inf')

        comparison = {
            "reef_per_seed_aurocs": {str(s): float(a) for s, a in zip(SEEDS, da_aurocs)},
            "reef_mean": round(da_mean, 6),
            "reef_sd": round(da_sd, 6),
            "mean_difference": round(float(da_mean - mean_auroc), 6),
            "paired_ttest_statistic": round(float(t_stat), 6),
            "t_pvalue": round(float(t_pvalue), 6),
            "cohens_d": round(cohens_d, 4),
            "sd_ratio_ablation_over_reef": round(sd_ratio, 4),
            "interpretation": (
                "SD ratio > 1.0 means ablation is less stable than REEF; "
                "SD ratio < 1.0 means ablation is more stable"
            ),
        }
        logger.info(f"\nComparison with REEF:")
        logger.info(f"  REEF mean AUROC: {da_mean:.4f} ± {da_sd:.4f}")
        logger.info(f"  Ablation mean AUROC:     {mean_auroc:.4f} ± {sd_auroc:.4f}")
        logger.info(f"  Mean difference (REEF - ablation): {da_mean - mean_auroc:+.4f}")
        logger.info(f"  Paired t-test p-value: {t_pvalue:.4f}")
        logger.info(f"  Cohen's d: {cohens_d:.4f}")
        logger.info(f"  SD ratio (ablation/DA): {sd_ratio:.4f}")
    else:
        logger.warning(f"  Main benchmark results not found at {main_results_path}")
        logger.warning("  Comparison statistics will not be computed.")

    # --- Step 6: Assemble and write results JSON ---
    results_json = {
        "ablation_name": "equal_weights",
        "description": (
            "REEF with equal topk ratios (0.65 each) and uniform "
            "embedding weights (1.0 each). Total features: 1300. All other "
            "parameters identical to main benchmark."
        ),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "omic_topk_ratios": ABLATION_CONFIG["omic_topk_ratios"],
            "omic_biological_weights": ABLATION_CONFIG["omic_biological_weights"],
            "topk_features": ABLATION_CONFIG["topk_features"],
            "n_rounds": ABLATION_CONFIG["n_rounds"],
            "hidden_dim": ABLATION_CONFIG["hidden_dim"],
            "embedding_dim": ABLATION_CONFIG["embedding_dim"],
            "lr_client": ABLATION_CONFIG["lr_client"],
            "lr_server": ABLATION_CONFIG["lr_server"],
            "weight_decay": ABLATION_CONFIG["weight_decay"],
            "dropout": ABLATION_CONFIG["dropout"],
            "patience": ABLATION_CONFIG["patience"],
            "batch_size": ABLATION_CONFIG["batch_size"],
            "domain_warmup_rounds": ABLATION_CONFIG["domain_warmup_rounds"],
            "total_features_selected": allocations["total"],
        },
        "seeds": SEEDS,
        "per_seed_auroc": per_seed_aurocs,
        "mean_auroc": round(mean_auroc, 6),
        "sd_auroc": round(sd_auroc, 6),
        "cv": round(cv, 6),
        "ci_95": [round(ci_95[0], 6), round(ci_95[1], 6)],
        "feature_allocations": allocations,
        "per_seed_saliency_rankings": per_seed_saliency,
        "saliency_dominance_summary": {
            "dominant_omic_frequency": dominance_counts,
            "note": f"Counts across all folds × seeds (max {13 * len(SEEDS)} = 13 folds × {len(SEEDS)} seeds)",
        },
        "comparison_with_reef": comparison,
        "per_seed_details": {
            str(s): {
                "auroc": per_seed_results[str(s)]["test_auroc"],
                "accuracy": per_seed_results[str(s)]["test_accuracy"],
                "f1": per_seed_results[str(s)]["test_f1"],
                "train_time": per_seed_results[str(s)]["train_time"],
                "avg_features_selected": per_seed_results[str(s)]["avg_features_selected"],
            }
            for s in SEEDS
        },
    }

    results_path = output_dir / "equal_weights_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # --- Step 7: Print summary ---
    logger.info(f"\n{'='*70}")
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  Mean AUROC:  {mean_auroc:.4f} ± {sd_auroc:.4f}")
    logger.info(f"  CV:          {cv:.4f}")
    logger.info(f"  95% CI:      [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    logger.info(f"  Features:    {allocations['total']} (equal allocation)")
    logger.info(f"  Per-seed:    {per_seed_aurocs}")
    logger.info(f"\n  Saliency dominance:")
    for omic, count in sorted(dominance_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {omic}: {count}/{13 * len(SEEDS)} folds")

    if comparison:
        logger.info(f"\n  Comparison with REEF:")
        logger.info(f"    Mean diff (REEF - ablation): {comparison['mean_difference']:+.4f}")
        logger.info(f"    p-value:    {comparison['t_pvalue']:.4f}")
        logger.info(f"    Cohen's d:  {comparison['cohens_d']:.4f}")
        logger.info(f"    SD ratio:   {comparison['sd_ratio_ablation_over_reef']:.4f}")


if __name__ == "__main__":
    main()
