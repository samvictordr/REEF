#!/usr/bin/env python3
"""
Comprehensive VFL Benchmark: Domain-Aware vs General VFL for Multi-Omics.

This benchmark compares three approaches for multi-omic stress classification:
1. NVFlare VFL (Base): Standard vertical federated learning using REAL NVFlare APIs
2. REEF: VFL with gradient saliency-based feature selection
3. LASER (General VFL): State-of-the-art general VFL baseline

IMPORTANT: This implementation uses REAL NVFlare APIs (DXO, Shareable patterns)
not simulated federated learning. See src/nvflare_vfl/nvflare_native_trainer.py

Purpose: Demonstrate that domain-aware VFL (saliency-guided) outperforms
general VFL approaches on small, high-dimensional multi-omic datasets.

Evaluation Modes:
- "holdout": Traditional 60/20/20 split (for quick testing)
- "loocv": Leave-One-Out Cross-Validation (recommended for N=13)

Metrics Computed:
- AUROC (primary), Accuracy, F1, Precision, Recall
- Training time, convergence speed
- Feature selection statistics
- Statistical significance (paired t-test, Wilcoxon)
- Effect sizes (Cohen's d)

For Research Paper:
- LOOCV for robust evaluation with small samples
- Multi-seed evaluation for robustness
- Ablation studies for hyperparameter sensitivity
- Per-omic contribution analysis
"""

import sys
import json
import random
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

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
    NVFlareNativeVFLTrainer, 
    NVFlareVFLConfig,
    DomainAwareNVFlareTrainer,
    DomainAwareNVFlareConfig,
    load_coral_data,
)
from benchmarks.adapters.laser_adapter import LASERAdapter, load_coral_data as load_laser_data
# Note: cv_utils provides LOOCVEvaluator and CVResults, but inline LOOCV is used here

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Metric Computation
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    auroc: float
    accuracy: float
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    train_time: float = 0.0
    n_rounds: int = 0
    convergence_round: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "auroc": self.auroc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "train_time": self.train_time,
            "n_rounds": self.n_rounds,
            "convergence_round": self.convergence_round,
        }


def compute_extended_metrics(
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute extended classification metrics."""
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, 
        precision_score, recall_score, brier_score_loss,
        average_precision_score, matthews_corrcoef
    )
    
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }
    
    # Handle single-class edge case
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = np.mean(y_true)
        metrics["mcc"] = 0.0
    
    return metrics


def compute_statistical_significance(
    results_a: List[float],
    results_b: List[float],
    method_a: str = "A",
    method_b: str = "B",
) -> Dict[str, Any]:
    """Compute statistical significance between two methods."""
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(results_a, results_b)
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(results_a, results_b)
    except ValueError:
        w_stat, w_pvalue = np.nan, 1.0
    
    # Effect size (Cohen's d)
    diff = results_a - results_b
    cohens_d = diff.mean() / (diff.std() + 1e-10)
    
    # Interpretation
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    return {
        "comparison": f"{method_a} vs {method_b}",
        "mean_diff": float(results_a.mean() - results_b.mean()),
        "paired_ttest": {"statistic": float(t_stat), "p_value": float(t_pvalue)},
        "wilcoxon": {"statistic": float(w_stat), "p_value": float(w_pvalue)},
        "cohens_d": float(cohens_d),
        "effect_interpretation": effect_interp,
        "significant_005": t_pvalue < 0.05,
        "significant_001": t_pvalue < 0.01,
    }


# =============================================================================
# Method Runners
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds."""
    random.seed(seed)  # FIX: seed Python random for LASER task sampling reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_nvflare_vfl(
    data_dir: Path,
    seed: int,
    n_rounds: int = 80,
) -> Dict[str, Any]:
    """Run base NVFlare VFL using REAL NVFlare APIs (DXO communication)."""
    logger.info(f"  Running NVFlare VFL (seed={seed}) - Real NVFlare APIs...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    # Load data using NVFlare-native loader
    data, labels, splits = load_coral_data(data_dir, seed=seed)
    
    # Create NVFlare-native config and trainer
    config = NVFlareVFLConfig(
        n_rounds=n_rounds,
        batch_size=8,
        lr_client=5e-5,
        lr_server=5e-4,
        patience=15,
        seed=seed,
    )
    
    trainer = NVFlareNativeVFLTrainer(data, labels, config)
    history = trainer.train(splits["train"], splits["val"])
    test_metrics = trainer.evaluate(splits["test"])
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "method": "nvflare_vfl",
        "implementation": "nvflare_native_dxo",  # Indicates real NVFlare APIs
        "seed": seed,
        "test_auroc": test_metrics["auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "best_val_auroc": trainer.best_val_auroc,
        "n_rounds_completed": len(history["train_loss"]),
        "train_time": train_time,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else np.nan,
    }


def run_reef(
    data_dir: Path,
    seed: int,
    n_rounds: int = 80,
    topk: int = 500,
    warmup_rounds: int = 20,
) -> Dict[str, Any]:
    """Run REEF using REAL NVFlare APIs (DXO communication)."""
    logger.info(f"  Running REEF VFL (seed={seed}, topk={topk}) - Real NVFlare APIs...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    # Load data using NVFlare-native loader
    data, labels, splits = load_coral_data(data_dir, seed=seed)
    
    # Create saliency-guided NVFlare config and trainer
    config = DomainAwareNVFlareConfig(
        n_rounds=n_rounds,
        topk_features=topk,
        domain_warmup_rounds=warmup_rounds,
        batch_size=8,
        lr_client=5e-5,
        lr_server=5e-4,
        patience=15,
        seed=seed,
    )
    
    trainer = DomainAwareNVFlareTrainer(data, labels, config)
    history = trainer.train(splits["train"], splits["val"])
    test_metrics = trainer.evaluate(splits["test"])
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Get feature selection stats
    total_original = sum(s["n_features"] for s in trainer.omic_stats.values())
    total_selected = sum(s["selected_features"] for s in trainer.omic_stats.values())
    
    return {
        "method": "reef",
        "implementation": "nvflare_native_dxo",  # Indicates real NVFlare APIs
        "seed": seed,
        "test_auroc": test_metrics["auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "best_val_auroc": trainer.best_val_auroc,
        "n_rounds_completed": history.get("total_rounds", n_rounds),
        "train_time": train_time,
        "features_original": total_original,
        "features_selected": total_selected,
        "feature_reduction": 1 - (total_selected / total_original),
        "omic_stats": trainer.omic_stats,
    }


def run_laser(
    data_dir: Path,
    seed: int,
    n_epochs: int = 80,
) -> Dict[str, Any]:
    """Run LASER baseline."""
    logger.info(f"  Running LASER (seed={seed})...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    try:
        data, sample_ids = load_laser_data(data_dir / "processed")
        labels = np.array([1 if "HiT" in s else 0 for s in sample_ids])
        omic_names = list(data.keys())
        
        config = {
            "hidden_dim": 128,
            "embedding_dim": 64,
            "n_epochs": n_epochs,
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "patience": 15,
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        adapter = LASERAdapter(
            data=data,
            labels=labels,
            omic_names=omic_names,
            config=config,
            device=device,
        )
        
        # Create splits
        n_samples = len(labels)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * n_samples)
        val_size = int(0.2 * n_samples)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        metrics = adapter.train(train_idx, val_idx, n_epochs=n_epochs, batch_size=8)
        
        test_loader = adapter.create_dataloader(test_idx, batch_size=8, shuffle=False, p_miss=0.0)
        test_metrics = adapter.evaluate(test_loader)
        
        train_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "method": "laser",
            "seed": seed,
            "test_auroc": test_metrics.get("auroc", 0.5),
            "test_accuracy": test_metrics.get("accuracy", 0.5),
            "best_val_auroc": metrics.get("best_val_auroc", 0.5),
            "n_epochs_completed": metrics.get("epochs_completed", n_epochs),
            "train_time": train_time,
        }
        
    except Exception as e:
        logger.error(f"  LASER failed: {e}")
        return {
            "method": "laser",
            "seed": seed,
            "test_auroc": 0.0,
            "test_accuracy": 0.0,
            "error": str(e),
            "train_time": 0.0,
        }


# =============================================================================
# LOOCV Runners (Recommended for N=13)
# =============================================================================

def run_nvflare_vfl_loocv(
    data_dir: Path,
    seed: int,
    n_rounds: int = 80,
    save_history: bool = False,
) -> Dict[str, Any]:
    """
    Run NVFlare VFL with Leave-One-Out Cross-Validation.
    
    With N=13, this gives 13 folds, each with 12 training and 1 test sample.
    Predictions are collected across all folds to compute overall AUROC.
    """
    logger.info(f"  Running NVFlare VFL LOOCV (seed={seed})...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    # Load data for LOOCV mode
    data, labels, splits = load_coral_data(data_dir, seed=seed, cv_mode="loocv")
    n_samples = splits["n_samples"]
    all_indices = splits["all_indices"]
    
    # Collect predictions from all folds
    all_preds = np.zeros(n_samples)
    all_labels = np.zeros(n_samples)
    fold_metrics = []
    representative_history = None
    
    for fold_idx in range(n_samples):
        # Leave fold_idx out for testing
        test_idx = np.array([fold_idx])
        train_idx = np.array([i for i in range(n_samples) if i != fold_idx])
        
        # For small sample, we use part of training as validation
        # Use a simple deterministic split: first sample as val
        if len(train_idx) > 2:
            val_idx = train_idx[:1]  # First sample as validation
            train_idx_actual = train_idx[1:]
        else:
            val_idx = train_idx[:1]
            train_idx_actual = train_idx
        
        config = NVFlareVFLConfig(
            n_rounds=n_rounds,
            batch_size=min(8, len(train_idx_actual)),
            lr_client=5e-5,
            lr_server=5e-4,
            patience=15,
            seed=seed + fold_idx,  # Vary seed per fold
        )
        
        trainer = NVFlareNativeVFLTrainer(data, labels, config)
        
        try:
            history = trainer.train(train_idx_actual, val_idx)
            test_metrics = trainer.evaluate(test_idx, return_predictions=True)
            
            # Capture training history from fold 0 for learning curve plots
            if save_history and fold_idx == 0 and representative_history is None:
                representative_history = history
            
            # Store prediction for this left-out sample
            preds = test_metrics.get("predictions", np.array([0.5]))
            all_preds[fold_idx] = float(preds[0]) if len(preds) > 0 else 0.5
            all_labels[fold_idx] = labels[fold_idx]
            
            fold_metrics.append({
                "fold": fold_idx,
                "prediction": all_preds[fold_idx],
                "label": int(labels[fold_idx]),
            })
        except Exception as e:
            logger.warning(f"    Fold {fold_idx} failed: {e}")
            all_preds[fold_idx] = 0.5
            all_labels[fold_idx] = labels[fold_idx]
    
    # Compute overall metrics across all LOOCV predictions
    overall_metrics = compute_extended_metrics(all_labels, all_preds)
    train_time = (datetime.now() - start_time).total_seconds()
    
    result = {
        "method": "nvflare_vfl",
        "cv_mode": "loocv",
        "seed": seed,
        "test_auroc": overall_metrics["auroc"],
        "test_accuracy": overall_metrics["accuracy"],
        "test_f1": overall_metrics.get("f1", 0.0),
        "n_folds": n_samples,
        "train_time": train_time,
        "fold_metrics": fold_metrics,
    }
    if representative_history is not None:
        result["training_history"] = representative_history
    return result


def run_reef_loocv(
    data_dir: Path,
    seed: int,
    n_rounds: int = 80,
    topk: int = 500,
    warmup_rounds: int = 20,
    save_history: bool = False,
) -> Dict[str, Any]:
    """
    Run REEF with Leave-One-Out Cross-Validation.
    """
    logger.info(f"  Running REEF VFL LOOCV (seed={seed}, topk={topk})...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    data, labels, splits = load_coral_data(data_dir, seed=seed, cv_mode="loocv")
    n_samples = splits["n_samples"]
    
    all_preds = np.zeros(n_samples)
    all_labels = np.zeros(n_samples)
    fold_metrics = []
    total_features_selected = []
    representative_history = None
    
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
            n_rounds=n_rounds,
            topk_features=topk,
            domain_warmup_rounds=warmup_rounds,
            batch_size=min(8, len(train_idx_actual)),
            lr_client=5e-5,
            lr_server=5e-4,
            patience=15,
            seed=seed + fold_idx,
        )
        
        trainer = DomainAwareNVFlareTrainer(data, labels, config)
        
        try:
            history = trainer.train(train_idx_actual, val_idx)
            test_metrics = trainer.evaluate(test_idx, return_predictions=True)
            
            # Capture training history from fold 0 for learning curve plots
            if save_history and fold_idx == 0 and representative_history is None:
                representative_history = history
            
            preds = test_metrics.get("predictions", np.array([0.5]))
            all_preds[fold_idx] = float(preds[0]) if len(preds) > 0 else 0.5
            all_labels[fold_idx] = labels[fold_idx]
            
            total_selected = sum(s.get("selected_features", 0) for s in trainer.omic_stats.values())
            total_features_selected.append(total_selected)
            
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
    
    overall_metrics = compute_extended_metrics(all_labels, all_preds)
    train_time = (datetime.now() - start_time).total_seconds()
    
    result = {
        "method": "reef",
        "cv_mode": "loocv",
        "seed": seed,
        "test_auroc": overall_metrics["auroc"],
        "test_accuracy": overall_metrics["accuracy"],
        "test_f1": overall_metrics.get("f1", 0.0),
        "n_folds": n_samples,
        "train_time": train_time,
        "avg_features_selected": float(np.mean(total_features_selected)) if total_features_selected else 0,
        "fold_metrics": fold_metrics,
    }
    if representative_history is not None:
        result["training_history"] = representative_history
    return result


def run_laser_loocv(
    data_dir: Path,
    seed: int,
    n_epochs: int = 80,
) -> Dict[str, Any]:
    """
    Run LASER with Leave-One-Out Cross-Validation.
    """
    logger.info(f"  Running LASER LOOCV (seed={seed})...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    try:
        data, sample_ids = load_laser_data(data_dir / "processed")
        labels = np.array([1 if "HiT" in s else 0 for s in sample_ids])
        omic_names = list(data.keys())
        n_samples = len(labels)
        
        config = {
            "hidden_dim": 128,
            "embedding_dim": 64,
            "n_epochs": n_epochs,
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "patience": 15,
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        all_preds = np.zeros(n_samples)
        all_labels = np.zeros(n_samples)
        fold_metrics = []
        
        for fold_idx in range(n_samples):
            test_idx = np.array([fold_idx])
            train_idx = np.array([i for i in range(n_samples) if i != fold_idx])
            
            if len(train_idx) > 2:
                val_idx = train_idx[:1]
                train_idx_actual = train_idx[1:]
            else:
                val_idx = train_idx[:1]
                train_idx_actual = train_idx
            
            adapter = LASERAdapter(
                data=data,
                labels=labels,
                omic_names=omic_names,
                config=config,
                device=device,
            )
            
            try:
                metrics = adapter.train(train_idx_actual, val_idx, n_epochs=n_epochs, batch_size=8)
                
                test_loader = adapter.create_dataloader(test_idx, batch_size=1, shuffle=False, p_miss=0.0)
                test_metrics = adapter.evaluate(test_loader, return_predictions=True)
                
                all_preds[fold_idx] = test_metrics.get("predictions", [0.5])[0] if test_metrics.get("predictions") is not None and len(test_metrics.get("predictions", [])) > 0 else 0.5
                all_labels[fold_idx] = labels[fold_idx]
                
                fold_metrics.append({
                    "fold": fold_idx,
                    "prediction": all_preds[fold_idx],
                    "label": int(labels[fold_idx]),
                })
            except Exception as e:
                logger.warning(f"    Fold {fold_idx} failed: {e}")
                all_preds[fold_idx] = 0.5
                all_labels[fold_idx] = labels[fold_idx]
        
        overall_metrics = compute_extended_metrics(all_labels, all_preds)
        train_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "method": "laser",
            "cv_mode": "loocv",
            "seed": seed,
            "test_auroc": overall_metrics["auroc"],
            "test_accuracy": overall_metrics["accuracy"],
            "test_f1": overall_metrics.get("f1", 0.0),
            "n_folds": n_samples,
            "train_time": train_time,
            "fold_metrics": fold_metrics,
        }
        
    except Exception as e:
        logger.error(f"  LASER LOOCV failed: {e}")
        return {
            "method": "laser",
            "cv_mode": "loocv",
            "seed": seed,
            "test_auroc": 0.0,
            "test_accuracy": 0.0,
            "error": str(e),
            "train_time": 0.0,
        }


# =============================================================================
# Negative Control (Shuffled Labels)
# =============================================================================

def run_negative_control(
    data_dir: Path,
    seed: int,
    n_rounds: int = 80,
    method: str = "reef",
) -> Dict[str, Any]:
    """
    Run negative control with shuffled labels.
    
    This validates that the model is learning from true signal, not noise.
    With shuffled labels, AUROC should be ~0.5 (random chance).
    
    If AUROC >> 0.5 with shuffled labels, it indicates:
    - Overfitting to noise
    - Data leakage
    - Model memorization
    
    Args:
        data_dir: Path to data directory
        seed: Random seed
        n_rounds: Training rounds
        method: Which method to test ("reef" or "nvflare_vfl")
    
    Returns:
        Dict with negative control results
    """
    logger.info(f"  Running Negative Control (shuffled labels, seed={seed})...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    # Load data
    data, labels, splits = load_coral_data(data_dir, seed=seed, cv_mode="loocv")
    n_samples = splits["n_samples"]
    
    # SHUFFLE LABELS - break the true association
    np.random.seed(seed + 9999)  # Different seed for shuffling
    shuffled_labels = labels.copy()
    np.random.shuffle(shuffled_labels)
    
    logger.info(f"    Original labels: {labels}")
    logger.info(f"    Shuffled labels: {shuffled_labels}")
    
    all_preds = np.zeros(n_samples)
    all_labels = np.zeros(n_samples)
    
    for fold_idx in range(n_samples):
        test_idx = np.array([fold_idx])
        train_idx = np.array([i for i in range(n_samples) if i != fold_idx])
        
        if len(train_idx) > 2:
            val_idx = train_idx[:1]
            train_idx_actual = train_idx[1:]
        else:
            val_idx = train_idx[:1]
            train_idx_actual = train_idx
        
        if method == "reef":
            config = DomainAwareNVFlareConfig(
                n_rounds=n_rounds,
                topk_features=500,
                domain_warmup_rounds=20,
                batch_size=min(8, len(train_idx_actual)),
                lr_client=5e-5,
                lr_server=5e-4,
                patience=15,
                seed=seed + fold_idx,
            )
            trainer = DomainAwareNVFlareTrainer(data, shuffled_labels, config)
        else:
            config = NVFlareVFLConfig(
                n_rounds=n_rounds,
                batch_size=min(8, len(train_idx_actual)),
                lr_client=5e-5,
                lr_server=5e-4,
                patience=15,
                seed=seed + fold_idx,
            )
            trainer = NVFlareNativeVFLTrainer(data, shuffled_labels, config)
        
        try:
            history = trainer.train(train_idx_actual, val_idx)
            test_metrics = trainer.evaluate(test_idx, return_predictions=True)
            
            preds = test_metrics.get("predictions", np.array([0.5]))
            all_preds[fold_idx] = float(preds[0]) if len(preds) > 0 else 0.5
            all_labels[fold_idx] = shuffled_labels[fold_idx]
        except Exception as e:
            logger.warning(f"    Fold {fold_idx} failed: {e}")
            all_preds[fold_idx] = 0.5
            all_labels[fold_idx] = shuffled_labels[fold_idx]
    
    overall_metrics = compute_extended_metrics(all_labels, all_preds)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Check if negative control is valid (AUROC should be ~0.5)
    auroc = overall_metrics["auroc"]
    is_valid = 0.3 <= auroc <= 0.7  # Allow some variance
    
    result = {
        "method": f"{method}_shuffled_labels",
        "control_type": "negative",
        "seed": seed,
        "test_auroc": auroc,
        "test_accuracy": overall_metrics["accuracy"],
        "is_valid_control": is_valid,
        "train_time": train_time,
        "interpretation": (
            "VALID: AUROC near chance level, model is not overfitting to noise"
            if is_valid else
            "WARNING: AUROC far from 0.5, possible overfitting or data leakage"
        ),
    }
    
    logger.info(f"    Negative control AUROC: {auroc:.4f} ({'VALID' if is_valid else 'WARNING'})")
    
    return result


def run_control_experiments(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    n_rounds: int = 80,
) -> Dict[str, Any]:
    """
    Run all control experiments for validation.
    
    Includes:
    1. Negative control (shuffled labels) - should give AUROC ~0.5
    2. (Future) Permutation importance validation
    """
    logger.info("=" * 70)
    logger.info("CONTROL EXPERIMENTS")
    logger.info("=" * 70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "controls": [],
    }
    
    # Negative control for REEF
    neg_domain = run_negative_control(data_dir, seed, n_rounds, method="reef")
    results["controls"].append(neg_domain)
    
    # Negative control for base VFL
    neg_base = run_negative_control(data_dir, seed, n_rounds, method="nvflare_vfl")
    results["controls"].append(neg_base)
    
    # Summary
    all_valid = all(c.get("is_valid_control", False) for c in results["controls"])
    results["all_controls_valid"] = all_valid
    results["summary"] = (
        "All negative controls passed - model is learning from true signal"
        if all_valid else
        "Some controls failed - investigate potential overfitting or data leakage"
    )
    
    logger.info(f"\n{results['summary']}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = output_dir / "control_experiments.json"
    with open(control_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Control results saved to {control_path}")
    
    return results


# =============================================================================
# Main Benchmark
# =============================================================================

def run_comprehensive_benchmark(
    data_dir: Path,
    output_dir: Path,
    seeds: List[int] = [42, 123, 456, 789, 1337],
    n_rounds: int = 80,
    laser_epochs: int = 30,
    topk: int = 500,
    cv_mode: str = "loocv",  # Default to LOOCV for small sample robustness
    run_ablation: bool = False,
    save_history: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark comparing all methods.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        seeds: Random seeds for reproducibility
        n_rounds: Number of VFL training rounds
        laser_epochs: Number of LASER training epochs
        topk: Top-k features for saliency-guided selection
        cv_mode: Evaluation mode - "loocv" (recommended) or "holdout"
        run_ablation: Whether to run ablation study
    
    Returns:
        Detailed results for research paper including:
        - Per-seed results for each method
        - Aggregate statistics (mean, std, CI)
        - Statistical significance tests
        - Effect sizes
    """
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE VFL BENCHMARK")
    logger.info("Domain-Aware vs General VFL for Multi-Omics")
    logger.info("=" * 70)
    logger.info(f"CV Mode: {cv_mode.upper()}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"VFL Rounds: {n_rounds}")
    logger.info(f"LASER Epochs: {laser_epochs}")
    logger.info(f"Domain saliency top-K: {topk}")
    
    if cv_mode == "loocv":
        logger.info("Using Leave-One-Out CV for robust small-sample evaluation")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": seeds,
            "n_rounds": n_rounds,
            "laser_epochs": laser_epochs,
            "topk": topk,
            "cv_mode": cv_mode,
        },
        "methods": {
            "nvflare_vfl": {"results": [], "aurocs": [], "accuracies": []},
            "reef": {"results": [], "aurocs": [], "accuracies": []},
            "laser": {"results": [], "aurocs": [], "accuracies": []},
        },
    }
    
    # Select runner functions based on cv_mode
    if cv_mode == "loocv":
        run_nvflare = run_nvflare_vfl_loocv
        run_domain = run_reef_loocv
        run_laser_fn = run_laser_loocv
    else:
        run_nvflare = run_nvflare_vfl
        run_domain = run_reef
        run_laser_fn = run_laser
    
    # Run each seed
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed}")
        logger.info(f"{'='*60}")
        
        # Save history only for the first seed to avoid bloating results
        save_this_seed = save_history and seed_idx == 0
        
        # NVFlare VFL
        nvflare_result = run_nvflare(data_dir, seed, n_rounds, save_history=save_this_seed) if cv_mode == "loocv" else run_nvflare(data_dir, seed, n_rounds)
        results["methods"]["nvflare_vfl"]["results"].append(nvflare_result)
        results["methods"]["nvflare_vfl"]["aurocs"].append(nvflare_result["test_auroc"])
        results["methods"]["nvflare_vfl"]["accuracies"].append(nvflare_result["test_accuracy"])
        logger.info(f"    NVFlare VFL AUROC: {nvflare_result['test_auroc']:.4f}")
        
        # REEF
        domain_result = run_domain(data_dir, seed, n_rounds, topk, save_history=save_this_seed) if cv_mode == "loocv" else run_domain(data_dir, seed, n_rounds, topk)
        results["methods"]["reef"]["results"].append(domain_result)
        results["methods"]["reef"]["aurocs"].append(domain_result["test_auroc"])
        results["methods"]["reef"]["accuracies"].append(domain_result["test_accuracy"])
        logger.info(f"    REEF AUROC: {domain_result['test_auroc']:.4f}")
        
        # LASER
        laser_result = run_laser_fn(data_dir, seed, laser_epochs)
        results["methods"]["laser"]["results"].append(laser_result)
        results["methods"]["laser"]["aurocs"].append(laser_result["test_auroc"])
        results["methods"]["laser"]["accuracies"].append(laser_result["test_accuracy"])
        logger.info(f"    LASER AUROC: {laser_result['test_auroc']:.4f}")
    
    # =========================================================================
    # Compute Aggregate Statistics
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'='*70}")
    
    for method_name, method_data in results["methods"].items():
        aurocs = np.array(method_data["aurocs"])
        accuracies = np.array(method_data["accuracies"])
        
        method_data["statistics"] = {
            "auroc": {
                "mean": float(aurocs.mean()),
                "std": float(aurocs.std()),
                "min": float(aurocs.min()),
                "max": float(aurocs.max()),
                "ci_95": [
                    float(aurocs.mean() - stats.t.ppf(0.975, len(aurocs) - 1) * aurocs.std() / np.sqrt(len(aurocs))),
                    float(aurocs.mean() + stats.t.ppf(0.975, len(aurocs) - 1) * aurocs.std() / np.sqrt(len(aurocs))),
                ],
            },
            "accuracy": {
                "mean": float(accuracies.mean()),
                "std": float(accuracies.std()),
            },
            "n_successful": int((aurocs > 0).sum()),
        }
        
        logger.info(
            f"{method_name:20s} | AUROC: {aurocs.mean():.4f} ± {aurocs.std():.4f} | "
            f"Acc: {accuracies.mean():.4f} ± {accuracies.std():.4f}"
        )
    
    # =========================================================================
    # Statistical Significance Tests
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("STATISTICAL SIGNIFICANCE")
    logger.info(f"{'='*70}")
    
    comparisons = [
        ("reef", "nvflare_vfl"),
        ("reef", "laser"),
        ("nvflare_vfl", "laser"),
    ]
    
    results["statistical_tests"] = {}
    
    for method_a, method_b in comparisons:
        aurocs_a = results["methods"][method_a]["aurocs"]
        aurocs_b = results["methods"][method_b]["aurocs"]
        
        sig_result = compute_statistical_significance(
            aurocs_a, aurocs_b, method_a, method_b
        )
        results["statistical_tests"][f"{method_a}_vs_{method_b}"] = sig_result
        
        logger.info(
            f"{method_a} vs {method_b}: "
            f"Δ={sig_result['mean_diff']:+.4f}, "
            f"p={sig_result['paired_ttest']['p_value']:.4f}, "
            f"d={sig_result['cohens_d']:.2f} ({sig_result['effect_interpretation']})"
        )
    
    # =========================================================================
    # Research Paper Summary
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("RESEARCH PAPER SUMMARY")
    logger.info(f"{'='*70}")
    
    domain_auroc = results["methods"]["reef"]["statistics"]["auroc"]["mean"]
    nvflare_auroc = results["methods"]["nvflare_vfl"]["statistics"]["auroc"]["mean"]
    laser_auroc = results["methods"]["laser"]["statistics"]["auroc"]["mean"]
    
    best_method = max(
        [("REEF", domain_auroc), ("NVFlare VFL", nvflare_auroc), ("LASER", laser_auroc)],
        key=lambda x: x[1]
    )
    
    results["summary"] = {
        "best_method": best_method[0],
        "best_auroc": best_method[1],
        "reef_improvement_over_base": domain_auroc - nvflare_auroc,
        "reef_improvement_over_laser": domain_auroc - laser_auroc,
        "reef_advantage": (domain_auroc + nvflare_auroc) / 2 - laser_auroc,
    }
    
    logger.info(f"Best Method: {best_method[0]} (AUROC: {best_method[1]:.4f})")
    logger.info(f"Domain saliency improvement over base VFL: {domain_auroc - nvflare_auroc:+.4f}")
    logger.info(f"Domain saliency improvement over LASER: {domain_auroc - laser_auroc:+.4f}")
    logger.info(f"Domain-aware advantage: {(domain_auroc + nvflare_auroc)/2 - laser_auroc:+.4f}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    results_path = output_dir / "comprehensive_benchmark.json"
    with open(results_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    # =========================================================================
    # Generate LaTeX Table
    # =========================================================================
    latex_table = generate_latex_table(results)
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    logger.info(f"LaTeX table saved to {latex_path}")
    
    return results


def generate_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table for research paper with LASER as baseline."""
    methods = results["methods"]
    
    # Find the best method by AUROC
    best_method = max(
        methods.keys(),
        key=lambda m: methods[m]["statistics"]["auroc"]["mean"]
    )
    
    table = r"""
\begin{table}[h]
\centering
\caption{Performance comparison of VFL methods on coral multi-omics stress classification (N=13 samples, P=90K features). LASER serves as the general VFL baseline. Results averaged over multiple random seeds.}
\label{tab:vfl_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{AUROC} & \textbf{Accuracy} & \textbf{vs LASER} & \textbf{p-value} \\
\midrule
"""
    
    laser_auroc = methods["laser"]["statistics"]["auroc"]["mean"]
    
    # Order: LASER (baseline), then NVFlare VFL, then REEF
    method_order = [
        ("laser", "LASER (Baseline)"),
        ("nvflare_vfl", "NVFlare VFL"),
        ("reef", "REEF"),
    ]
    
    for method_name, display_name in method_order:
        stats = methods[method_name]["statistics"]
        auroc_mean = stats["auroc"]["mean"]
        auroc_std = stats["auroc"]["std"]
        acc_mean = stats["accuracy"]["mean"]
        acc_std = stats["accuracy"]["std"]
        
        improvement = auroc_mean - laser_auroc
        
        # Get p-value vs LASER
        if method_name != "laser":
            test_key = f"{method_name}_vs_laser"
            p_val = results["statistical_tests"].get(test_key, {}).get(
                "paired_ttest", {}
            ).get("p_value", 1.0)
            p_str = f"{p_val:.3f}" if p_val else "-"
            imp_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
        else:
            p_str = "-"
            imp_str = "(baseline)"
        
        # Bold the best performing method
        if method_name == best_method:
            table += f"\\textbf{{{display_name}}} & \\textbf{{{auroc_mean:.3f} $\\pm$ {auroc_std:.3f}}} & \\textbf{{{acc_mean:.3f} $\\pm$ {acc_std:.3f}}} & {imp_str} & {p_str} \\\\\n"
        else:
            table += f"{display_name} & {auroc_mean:.3f} $\\pm$ {auroc_std:.3f} & {acc_mean:.3f} $\\pm$ {acc_std:.3f} & {imp_str} & {p_str} \\\\\n"
    
    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return table


# =============================================================================
# Ablation Study
# =============================================================================

def run_ablation_comparison(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run ablation study comparing domain saliency hyperparameters."""
    from src.federated.domain_aware_vfl import run_ablation_study
    
    logger.info("=" * 70)
    logger.info("ABLATION STUDY")
    logger.info("=" * 70)
    
    results = run_ablation_study(
        data_dir=data_dir,
        output_dir=output_dir / "ablation",
        seed=seed,
        topk_values=[100, 250, 500, 1000],
        warmup_values=[10, 20, 30],
    )
    
    return results


# =============================================================================
# Missing Modality Ablation Study
# =============================================================================

def run_missing_modality_ablation(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    n_rounds: int = 50,
    laser_epochs: int = 30,
) -> Dict[str, Any]:
    """
    Run missing modality ablation study.
    
    Tests model robustness when each omic modality is dropped.
    This addresses the P≫N concern by measuring how dependent 
    the model is on each data source.
    
    Returns:
        Dict with results for each modality exclusion scenario.
    """
    logger.info("=" * 70)
    logger.info("MISSING MODALITY ABLATION STUDY")
    logger.info("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # All available modalities
    all_modalities = ["transcriptomics", "proteomics", "metabolomics", "microbiome"]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "full_model": {},
        "modality_ablations": {},
    }
    
    # First, run with all modalities (baseline)
    logger.info("\n--- Baseline: All modalities ---")
    try:
        baseline_result = run_nvflare_vfl_loocv(data_dir, seed, n_rounds)
        results["full_model"] = {
            "modalities": all_modalities,
            "auroc": baseline_result["test_auroc"],
            "accuracy": baseline_result["test_accuracy"],
        }
        logger.info(f"  All modalities AUROC: {baseline_result['test_auroc']:.4f}")
    except Exception as e:
        logger.warning(f"  Baseline failed: {e}")
        results["full_model"] = {"error": str(e)}
    
    # Run leave-one-modality-out ablations
    for exclude_modality in all_modalities:
        logger.info(f"\n--- Excluding: {exclude_modality} ---")
        included = [m for m in all_modalities if m != exclude_modality]
        
        try:
            # Load data with modality exclusion
            data, labels, splits = load_coral_data_with_exclusion(
                data_dir, 
                seed=seed, 
                exclude_modalities=[exclude_modality],
                cv_mode="loocv"
            )
            
            n_samples = splits["n_samples"]
            all_preds = np.zeros(n_samples)
            all_labels = np.zeros(n_samples)
            
            for fold_idx in range(n_samples):
                test_idx = np.array([fold_idx])
                train_idx = np.array([i for i in range(n_samples) if i != fold_idx])
                
                if len(train_idx) > 2:
                    val_idx = train_idx[:1]
                    train_idx_actual = train_idx[1:]
                else:
                    val_idx = train_idx[:1]
                    train_idx_actual = train_idx
                
                config = NVFlareVFLConfig(
                    n_rounds=n_rounds,
                    batch_size=min(8, len(train_idx_actual)),
                    lr_client=5e-5,
                    lr_server=5e-4,
                    patience=15,
                    seed=seed + fold_idx,
                )
                
                trainer = NVFlareNativeVFLTrainer(data, labels, config)
                
                try:
                    trainer.train(train_idx_actual, val_idx)
                    test_metrics = trainer.evaluate(test_idx, return_predictions=True)
                    preds = test_metrics.get("predictions", np.array([0.5]))
                    all_preds[fold_idx] = float(preds[0]) if len(preds) > 0 else 0.5
                except Exception:
                    all_preds[fold_idx] = 0.5
                    
                all_labels[fold_idx] = labels[fold_idx]
            
            overall_metrics = compute_extended_metrics(all_labels, all_preds)
            
            results["modality_ablations"][exclude_modality] = {
                "excluded": exclude_modality,
                "included": included,
                "auroc": overall_metrics["auroc"],
                "accuracy": overall_metrics["accuracy"],
                "auroc_drop": results["full_model"].get("auroc", 0.5) - overall_metrics["auroc"],
            }
            logger.info(f"  Without {exclude_modality}: AUROC={overall_metrics['auroc']:.4f}")
            
        except Exception as e:
            logger.warning(f"  Ablation for {exclude_modality} failed: {e}")
            results["modality_ablations"][exclude_modality] = {"error": str(e)}
    
    # Summary
    logger.info("\n--- Missing Modality Ablation Summary ---")
    if "auroc" in results["full_model"]:
        logger.info(f"  Full model AUROC: {results['full_model']['auroc']:.4f}")
    
    for modality, data in results["modality_ablations"].items():
        if "auroc" in data:
            drop = data.get("auroc_drop", 0)
            logger.info(f"  Without {modality}: AUROC={data['auroc']:.4f} (drop: {drop:+.4f})")
    
    # Save results
    output_path = output_dir / "missing_modality_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    
    return results


def load_coral_data_with_exclusion(
    data_dir: Path,
    seed: int = 42,
    exclude_modalities: List[str] = None,
    cv_mode: str = "loocv",
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """
    Load coral data with optional modality exclusion for ablation studies.
    
    Args:
        data_dir: Path to data directory
        seed: Random seed
        exclude_modalities: List of modality names to exclude
        cv_mode: Cross-validation mode
    
    Returns:
        Tuple of (data dict, labels, splits dict)
    """
    import pandas as pd
    
    exclude_modalities = exclude_modalities or []
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    
    if not processed_dir.exists():
        processed_dir = data_dir
    
    omic_file_patterns = {
        "transcriptomics": ["transcriptomics_tpm.tsv", "transcriptomics.tsv"],
        "proteomics": ["proteomics_imputed.tsv", "proteomics.tsv"], 
        "metabolomics": ["metabolomics_filtered.tsv", "metabolomics.tsv"],
        "microbiome": ["microbiome_filtered.tsv", "microbiome.tsv"],
    }
    
    # Filter out excluded modalities
    included_omics = {k: v for k, v in omic_file_patterns.items() 
                      if k not in exclude_modalities}
    
    dfs = {}
    all_sample_sets = []
    
    for omic_name, filenames in included_omics.items():
        for filename in filenames:
            filepath = processed_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, sep="\t", index_col=0)
                dfs[omic_name] = df
                all_sample_sets.append(set(df.index.tolist()))
                logger.debug(f"Loaded {omic_name}: {df.shape}")
                break
    
    if not dfs:
        raise ValueError(f"No omics data found (excluding {exclude_modalities})")
    
    # Find common samples
    if len(all_sample_sets) > 1:
        common_samples = set.intersection(*all_sample_sets)
    else:
        common_samples = all_sample_sets[0]
    
    common_samples = sorted(list(common_samples))
    n_samples = len(common_samples)
    
    # Extract labels from sample names
    labels = np.array([
        0 if "Amb" in s else 1 for s in common_samples
    ])
    
    # Build data dict
    data = {}
    for omic_name, df in dfs.items():
        aligned_df = df.loc[common_samples]
        data[omic_name] = aligned_df.values.astype(np.float32)
        # Handle NaN values
        data[omic_name] = np.nan_to_num(data[omic_name], nan=0.0)
    
    # Create splits
    np.random.seed(seed)
    all_indices = np.arange(n_samples)
    
    splits = {
        "n_samples": n_samples,
        "all_indices": all_indices,
        "sample_ids": common_samples,
    }
    
    return data, labels, splits


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive VFL Benchmark for Multi-Omics"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1337")
    parser.add_argument("--n-rounds", type=int, default=80)
    parser.add_argument("--laser-epochs", type=int, default=30, help="Number of epochs for LASER (default: 30)")
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument(
        "--cv-mode", 
        type=str, 
        default="loocv", 
        choices=["loocv", "holdout"],
        help="Cross-validation mode: 'loocv' (Leave-One-Out, recommended for N=13) or 'holdout' (60/20/20 split)"
    )
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--controls", action="store_true", help="Run negative control experiments with shuffled labels")
    parser.add_argument("--quick", action="store_true", help="Quick run with 2 seeds")
    parser.add_argument("--save-history", action="store_true", help="Save training history (loss/AUROC per round) for learning curve plots")
    parser.add_argument(
        "--missing-modality",
        action="store_true",
        help="Run missing modality ablation study (leave-one-modality-out)"
    )
    args = parser.parse_args()
    
    project_root = PROJECT_ROOT
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        seeds = seeds[:2]
    
    # Track whether any sub-task flags were given
    sub_task_only = args.controls or args.ablation or args.missing_modality

    # Run control experiments if requested
    if args.controls:
        run_control_experiments(data_dir, output_dir, seed=seeds[0], n_rounds=args.n_rounds)
    
    # Run ablation if requested
    if args.ablation:
        run_ablation_comparison(data_dir, output_dir, seed=seeds[0])
    
    # Run missing modality ablation if requested
    if args.missing_modality:
        run_missing_modality_ablation(
            data_dir, 
            output_dir / "missing_modality", 
            seed=seeds[0],
            n_rounds=args.n_rounds,
            laser_epochs=args.laser_epochs,
        )
    
    # Only run the full benchmark if no sub-task flags were given
    if sub_task_only:
        return None

    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(
        data_dir=data_dir,
        output_dir=output_dir,
        seeds=seeds,
        n_rounds=args.n_rounds,
        laser_epochs=args.laser_epochs,
        topk=args.topk,
        cv_mode=args.cv_mode,
        save_history=args.save_history,
    )
    
    return results


if __name__ == "__main__":
    main()
