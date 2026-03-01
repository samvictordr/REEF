#!/usr/bin/env python3
"""
Cross-Validation Utilities for Small-Sample Multi-Omics Benchmarks.

This module implements Leave-One-Out Cross-Validation (LOOCV) and nested 
cross-validation strategies appropriate for the extreme P≫N regime 
(N=13 samples, P≈90,000 features).

For N=13, traditional train/val/test splits are problematic:
- 60/20/20 split → 7/2/4 samples (2-sample validation is unreliable)
- LOOCV provides more robust evaluation for small N

Usage:
    from benchmarks.cv_utils import LOOCVEvaluator, NestedCVEvaluator
    
    evaluator = LOOCVEvaluator(data, labels, omic_names, model_factory)
    results = evaluator.run(n_epochs=100, early_stopping_patience=15)
    
    # Results contain per-fold metrics and aggregated statistics
    print(f"Mean AUROC: {results['aggregate']['auroc']['mean']:.3f}")
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import json

from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CVFoldResult:
    """Result from a single cross-validation fold."""
    fold_idx: int
    test_sample_idx: int
    test_sample_id: str
    true_label: int
    predicted_prob: float
    predicted_label: int
    train_indices: List[int]
    
    # Per-fold metrics (single sample, so limited)
    correct: bool = field(init=False)
    
    def __post_init__(self):
        self.correct = (self.true_label == self.predicted_label)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "test_sample_idx": self.test_sample_idx,
            "test_sample_id": self.test_sample_id,
            "true_label": int(self.true_label),
            "predicted_prob": float(self.predicted_prob),
            "predicted_label": int(self.predicted_label),
            "correct": self.correct,
        }


@dataclass
class CVResults:
    """Aggregated cross-validation results."""
    method_name: str
    cv_type: str  # "LOOCV", "StratifiedKFold", etc.
    n_folds: int
    n_samples: int
    
    fold_results: List[CVFoldResult]
    
    # Aggregated predictions (for overall metrics)
    y_true: np.ndarray = field(default=None, repr=False)
    y_prob: np.ndarray = field(default=None, repr=False)
    y_pred: np.ndarray = field(default=None, repr=False)
    
    # Aggregated metrics
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    
    def compute_aggregate_metrics(self):
        """Compute overall metrics from all fold predictions."""
        if self.y_true is None:
            self.y_true = np.array([r.true_label for r in self.fold_results])
            self.y_prob = np.array([r.predicted_prob for r in self.fold_results])
            self.y_pred = np.array([r.predicted_label for r in self.fold_results])
        
        metrics = {
            "accuracy": float(accuracy_score(self.y_true, self.y_pred)),
            "f1": float(f1_score(self.y_true, self.y_pred, zero_division=0)),
            "n_correct": int(sum(r.correct for r in self.fold_results)),
        }
        
        # AUROC requires both classes in the test set (which we have in LOOCV)
        if len(np.unique(self.y_true)) > 1:
            metrics["auroc"] = float(roc_auc_score(self.y_true, self.y_prob))
        else:
            metrics["auroc"] = 0.5
            
        # Standard error for LOOCV accuracy
        metrics["accuracy_se"] = float(np.sqrt(
            metrics["accuracy"] * (1 - metrics["accuracy"]) / self.n_samples
        ))
        
        self.aggregate_metrics = metrics
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        if not self.aggregate_metrics:
            self.compute_aggregate_metrics()
            
        return {
            "method_name": self.method_name,
            "cv_type": self.cv_type,
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
            "aggregate_metrics": self.aggregate_metrics,
            "per_fold_results": [r.to_dict() for r in self.fold_results],
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        if not self.aggregate_metrics:
            self.compute_aggregate_metrics()
            
        m = self.aggregate_metrics
        lines = [
            f"=== {self.method_name} - {self.cv_type} Results ===",
            f"Samples: {self.n_samples}, Folds: {self.n_folds}",
            f"",
            f"Aggregate Metrics:",
            f"  AUROC:    {m.get('auroc', 'N/A'):.3f}",
            f"  Accuracy: {m.get('accuracy', 'N/A'):.3f} ± {m.get('accuracy_se', 0):.3f}",
            f"  F1 Score: {m.get('f1', 'N/A'):.3f}",
            f"  Correct:  {m.get('n_correct', 0)}/{self.n_samples}",
            f"",
            f"Per-Fold Breakdown:",
        ]
        
        for r in self.fold_results:
            status = "✓" if r.correct else "✗"
            lines.append(
                f"  Fold {r.fold_idx:2d}: {r.test_sample_id[:20]:20s} "
                f"true={r.true_label} pred={r.predicted_label} "
                f"prob={r.predicted_prob:.3f} {status}"
            )
        
        return "\n".join(lines)


class LOOCVEvaluator:
    """
    Leave-One-Out Cross-Validation evaluator for VFL methods.
    
    For each fold:
    1. Leave out one sample as test
    2. Train on remaining N-1 samples
    3. Predict on held-out sample
    4. Aggregate predictions across all folds
    
    This provides the most robust evaluation for small N, as:
    - Each sample appears exactly once as test
    - Training uses maximum data (N-1 samples)
    - No arbitrary split decisions affect results
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        labels: np.ndarray,
        sample_ids: List[str],
        omic_names: List[str],
        model_factory: Callable,
        device: torch.device = None,
    ):
        """
        Initialize LOOCV evaluator.
        
        Args:
            data: Dict mapping omic names to (n_samples, n_features) arrays
            labels: Binary labels array
            sample_ids: List of sample identifiers (for reporting)
            omic_names: Ordered list of omics names
            model_factory: Callable that creates a fresh model for each fold
            device: Torch device
        """
        self.data = data
        self.labels = labels
        self.sample_ids = sample_ids
        self.omic_names = omic_names
        self.model_factory = model_factory
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.n_samples = len(labels)
        self.loo = LeaveOneOut()
        
    def run(
        self,
        method_name: str = "VFL",
        n_epochs: int = 100,
        early_stopping_patience: int = 15,
        verbose: bool = True,
    ) -> CVResults:
        """
        Run LOOCV evaluation.
        
        Args:
            method_name: Name of the method (for reporting)
            n_epochs: Maximum training epochs per fold
            early_stopping_patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            CVResults with per-fold and aggregate metrics
        """
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.loo.split(self.labels)):
            if verbose:
                logger.info(f"LOOCV Fold {fold_idx + 1}/{self.n_samples}: "
                           f"Test sample = {self.sample_ids[test_idx[0]]}")
            
            # Fit scalers on training data only
            scalers = {}
            scaled_data = {}
            for name, arr in self.data.items():
                scaler = StandardScaler()
                train_data = arr[train_idx]
                scaler.fit(train_data)
                scaled_data[name] = scaler.transform(arr)
                scalers[name] = scaler
            
            # Create fresh model for this fold
            model = self.model_factory()
            
            # Train the model (using all training data, no validation split)
            # For LOOCV, we typically use early stopping based on training loss
            # or a fixed number of epochs
            train_preds = self._train_fold(
                model=model,
                scaled_data=scaled_data,
                train_idx=train_idx,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
            )
            
            # Predict on test sample
            test_pred = self._predict_fold(
                model=model,
                scaled_data=scaled_data,
                test_idx=test_idx,
            )
            
            # Store result
            result = CVFoldResult(
                fold_idx=fold_idx,
                test_sample_idx=int(test_idx[0]),
                test_sample_id=self.sample_ids[test_idx[0]],
                true_label=int(self.labels[test_idx[0]]),
                predicted_prob=float(test_pred[0]),
                predicted_label=int(test_pred[0] >= 0.5),
                train_indices=train_idx.tolist(),
            )
            fold_results.append(result)
            
            if verbose:
                status = "✓" if result.correct else "✗"
                logger.info(f"  Result: true={result.true_label}, "
                           f"pred={result.predicted_label}, "
                           f"prob={result.predicted_prob:.3f} {status}")
        
        # Aggregate results
        cv_results = CVResults(
            method_name=method_name,
            cv_type="LOOCV",
            n_folds=self.n_samples,
            n_samples=self.n_samples,
            fold_results=fold_results,
        )
        cv_results.compute_aggregate_metrics()
        
        if verbose:
            logger.info(cv_results.summary())
        
        return cv_results
    
    def _train_fold(
        self,
        model: nn.Module,
        scaled_data: Dict[str, np.ndarray],
        train_idx: np.ndarray,
        n_epochs: int,
        early_stopping_patience: int,
    ) -> np.ndarray:
        """Train model on fold training data."""
        # This should be implemented by subclasses for specific model types
        raise NotImplementedError("Subclasses must implement _train_fold")
    
    def _predict_fold(
        self,
        model: nn.Module,
        scaled_data: Dict[str, np.ndarray],
        test_idx: np.ndarray,
    ) -> np.ndarray:
        """Predict on fold test data."""
        # This should be implemented by subclasses for specific model types
        raise NotImplementedError("Subclasses must implement _predict_fold")


class StratifiedKFoldEvaluator:
    """
    Stratified K-Fold Cross-Validation evaluator.
    
    For small N with class imbalance, stratified K-fold ensures each fold
    has approximately the same class distribution.
    
    Recommended for N ≥ 10 with k = 5 (2 test samples per fold).
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        labels: np.ndarray,
        sample_ids: List[str],
        omic_names: List[str],
        model_factory: Callable,
        n_splits: int = 5,
        device: torch.device = None,
    ):
        """
        Initialize Stratified K-Fold evaluator.
        
        Args:
            data: Dict mapping omic names to feature arrays
            labels: Binary labels array
            sample_ids: List of sample identifiers
            omic_names: Ordered list of omics names
            model_factory: Callable that creates a fresh model
            n_splits: Number of folds (default 5)
            device: Torch device
        """
        self.data = data
        self.labels = labels
        self.sample_ids = sample_ids
        self.omic_names = omic_names
        self.model_factory = model_factory
        self.n_splits = n_splits
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.n_samples = len(labels)
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def run(
        self,
        method_name: str = "VFL",
        n_epochs: int = 100,
        early_stopping_patience: int = 15,
        verbose: bool = True,
    ) -> CVResults:
        """Run stratified K-fold evaluation."""
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.skf.split(np.zeros(self.n_samples), self.labels)
        ):
            if verbose:
                test_ids = [self.sample_ids[i] for i in test_idx]
                logger.info(f"Fold {fold_idx + 1}/{self.n_splits}: "
                           f"Test samples = {test_ids}")
            
            # Fit scalers on training data only
            scalers = {}
            scaled_data = {}
            for name, arr in self.data.items():
                scaler = StandardScaler()
                train_data = arr[train_idx]
                scaler.fit(train_data)
                scaled_data[name] = scaler.transform(arr)
                scalers[name] = scaler
            
            # Create fresh model
            model = self.model_factory()
            
            # Train
            self._train_fold(
                model=model,
                scaled_data=scaled_data,
                train_idx=train_idx,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
            )
            
            # Predict on all test samples
            test_preds = self._predict_fold(
                model=model,
                scaled_data=scaled_data,
                test_idx=test_idx,
            )
            
            # Store results for each test sample
            for i, idx in enumerate(test_idx):
                result = CVFoldResult(
                    fold_idx=fold_idx,
                    test_sample_idx=int(idx),
                    test_sample_id=self.sample_ids[idx],
                    true_label=int(self.labels[idx]),
                    predicted_prob=float(test_preds[i]),
                    predicted_label=int(test_preds[i] >= 0.5),
                    train_indices=train_idx.tolist(),
                )
                fold_results.append(result)
        
        cv_results = CVResults(
            method_name=method_name,
            cv_type=f"StratifiedKFold-{self.n_splits}",
            n_folds=self.n_splits,
            n_samples=self.n_samples,
            fold_results=fold_results,
        )
        cv_results.compute_aggregate_metrics()
        
        if verbose:
            logger.info(cv_results.summary())
        
        return cv_results
    
    def _train_fold(self, model, scaled_data, train_idx, n_epochs, early_stopping_patience):
        raise NotImplementedError("Subclasses must implement _train_fold")
    
    def _predict_fold(self, model, scaled_data, test_idx):
        raise NotImplementedError("Subclasses must implement _predict_fold")


def compare_cv_results(
    results_list: List[CVResults],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare CV results across multiple methods.
    
    Args:
        results_list: List of CVResults from different methods
        output_path: Optional path to save comparison JSON
        
    Returns:
        Comparison summary dict
    """
    comparison = {
        "methods": [],
        "metrics_comparison": {},
    }
    
    for results in results_list:
        if not results.aggregate_metrics:
            results.compute_aggregate_metrics()
        
        comparison["methods"].append({
            "name": results.method_name,
            "cv_type": results.cv_type,
            "metrics": results.aggregate_metrics,
        })
    
    # Compute pairwise comparisons
    metric_names = ["auroc", "accuracy", "f1"]
    for metric in metric_names:
        comparison["metrics_comparison"][metric] = {}
        values = [r.aggregate_metrics.get(metric, 0.0) for r in results_list]
        names = [r.method_name for r in results_list]
        
        best_idx = np.argmax(values)
        comparison["metrics_comparison"][metric] = {
            "values": dict(zip(names, values)),
            "best_method": names[best_idx],
            "best_value": values[best_idx],
        }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved CV comparison to {output_path}")
    
    return comparison


def print_cv_summary_table(results_list: List[CVResults]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    
    # Header
    print(f"{'Method':<25} {'CV Type':<15} {'AUROC':>8} {'Accuracy':>10} {'F1':>8}")
    print("-" * 70)
    
    for r in results_list:
        if not r.aggregate_metrics:
            r.compute_aggregate_metrics()
        m = r.aggregate_metrics
        print(f"{r.method_name:<25} {r.cv_type:<15} "
              f"{m.get('auroc', 0):.3f}    "
              f"{m.get('accuracy', 0):.3f}±{m.get('accuracy_se', 0):.3f}  "
              f"{m.get('f1', 0):.3f}")
    
    print("=" * 70)
