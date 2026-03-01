"""
Evaluation metrics for coral multi-omics models.

Provides:
- Classification metrics (AUROC, AUPRC, accuracy, F1, etc.)
- Regression metrics (MSE, MAE, R², Pearson/Spearman correlation)
- MetricsTracker for logging across folds/experiments
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr


logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    specificity: float
    balanced_accuracy: float
    
    # For plotting
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    precision_curve: Optional[np.ndarray] = None
    recall_curve: Optional[np.ndarray] = None
    
    # Confusion matrix
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    
    def to_dict(self, include_curves: bool = False) -> Dict:
        """Convert to dict (optionally excluding curves)."""
        d = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }
        if include_curves and self.fpr is not None:
            d["fpr"] = self.fpr.tolist()
            d["tpr"] = self.tpr.tolist()
        if include_curves and self.precision_curve is not None:
            d["precision_curve"] = self.precision_curve.tolist()
            d["recall_curve"] = self.recall_curve.tolist()
        return d


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    
    mse: float
    rmse: float
    mae: float
    r2: float
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (0/1)
        y_pred: Predicted labels (0/1) or probabilities
        y_prob: Predicted probabilities (if y_pred is labels)
        threshold: Threshold for converting probabilities to labels
    
    Returns:
        ClassificationMetrics instance
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Determine if y_pred is probabilities or labels
    if y_prob is None:
        if np.any((y_pred > 0) & (y_pred < 1)):
            # y_pred is probabilities
            y_prob = y_pred
            y_pred = (y_prob >= threshold).astype(int)
        else:
            # y_pred is labels, no probabilities available
            y_prob = y_pred  # Use labels as pseudo-probabilities
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity and balanced accuracy
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    # AUC metrics
    try:
        auroc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError:
        auroc = 0.5
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
    
    try:
        auprc = average_precision_score(y_true, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    except ValueError:
        auprc = np.mean(y_true)
        precision_curve, recall_curve = None, None
    
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
        auprc=auprc,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        fpr=fpr,
        tpr=tpr,
        precision_curve=precision_curve,
        recall_curve=recall_curve,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """
    Compute comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RegressionMetrics instance
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
    )


class MetricsTracker:
    """
    Track and aggregate metrics across folds/experiments.
    
    Supports:
    - Per-fold metric storage
    - Aggregation (mean, std, CI)
    - Export to JSON/CSV
    """
    
    def __init__(
        self,
        experiment_name: str,
        task: str = "classification",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize tracker.
        
        Args:
            experiment_name: Name of the experiment
            task: 'classification' or 'regression'
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.task = task
        self.output_dir = Path(output_dir) if output_dir else None
        
        self.folds: List[Dict] = []
        self.metadata: Dict = {
            "experiment_name": experiment_name,
            "task": task,
            "created_at": datetime.now().isoformat(),
        }
    
    def add_fold(
        self,
        fold_idx: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        additional_info: Optional[Dict] = None,
    ):
        """
        Add results for a fold.
        
        Args:
            fold_idx: Fold index
            y_true: True labels/values
            y_pred: Predicted labels/values or probabilities
            y_prob: Predicted probabilities (for classification)
            additional_info: Additional info to store
        """
        if self.task == "classification":
            metrics = compute_classification_metrics(y_true, y_pred, y_prob)
            metrics_dict = metrics.to_dict(include_curves=False)
        else:
            metrics = compute_regression_metrics(y_true, y_pred)
            metrics_dict = metrics.to_dict()
        
        fold_data = {
            "fold_idx": fold_idx,
            "n_samples": len(y_true),
            "metrics": metrics_dict,
        }
        
        if additional_info:
            fold_data["additional_info"] = additional_info
        
        self.folds.append(fold_data)
        
        logger.info(f"Fold {fold_idx}: {self._format_metrics(metrics_dict)}")
    
    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for logging."""
        if self.task == "classification":
            return f"AUROC={metrics['auroc']:.3f}, F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}"
        else:
            return f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}"
    
    def get_aggregated(self) -> Dict:
        """
        Get aggregated metrics across folds.
        
        Returns:
            Dict with mean, std, and 95% CI for each metric
        """
        if not self.folds:
            return {}
        
        # Collect metric values
        metric_names = list(self.folds[0]["metrics"].keys())
        values = {name: [] for name in metric_names}
        
        for fold in self.folds:
            for name in metric_names:
                values[name].append(fold["metrics"][name])
        
        # Compute aggregates
        aggregated = {}
        for name, vals in values.items():
            vals = np.array(vals)
            mean = np.mean(vals)
            std = np.std(vals)
            ci_lower = np.percentile(vals, 2.5)
            ci_upper = np.percentile(vals, 97.5)
            
            aggregated[name] = {
                "mean": float(mean),
                "std": float(std),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "n_folds": len(vals),
            }
        
        return aggregated
    
    def get_summary_table(self) -> str:
        """Get formatted summary table."""
        agg = self.get_aggregated()
        if not agg:
            return "No results yet."
        
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Task: {self.task}",
            f"Folds: {len(self.folds)}",
            "",
            "Metric".ljust(20) + "Mean ± Std".ljust(20) + "95% CI",
            "-" * 60,
        ]
        
        for name, stats in agg.items():
            mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            ci = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            lines.append(f"{name.ljust(20)}{mean_std.ljust(20)}{ci}")
        
        return "\n".join(lines)
    
    def save(self, filename: Optional[str] = None):
        """
        Save results to JSON file.
        
        Args:
            filename: Optional filename (default: experiment_name.json)
        """
        if self.output_dir is None:
            logger.warning("No output directory specified, skipping save")
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = filename or f"{self.experiment_name}.json"
        filepath = self.output_dir / filename
        
        data = {
            "metadata": self.metadata,
            "folds": self.folds,
            "aggregated": self.get_aggregated(),
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")
    
    def save_csv(self, filename: Optional[str] = None):
        """Save aggregated results to CSV."""
        import pandas as pd
        
        if self.output_dir is None:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = filename or f"{self.experiment_name}.csv"
        filepath = self.output_dir / filename
        
        agg = self.get_aggregated()
        rows = []
        for name, stats in agg.items():
            rows.append({
                "metric": name,
                "mean": stats["mean"],
                "std": stats["std"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved CSV to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MetricsTracker":
        """Load tracker from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        tracker = cls(
            experiment_name=data["metadata"]["experiment_name"],
            task=data["metadata"]["task"],
        )
        tracker.metadata = data["metadata"]
        tracker.folds = data["folds"]
        
        return tracker


def aggregate_experiment_results(
    result_files: List[Union[str, Path]],
    output_path: Union[str, Path],
) -> Dict:
    """
    Aggregate results from multiple experiment files into a comparison table.
    
    Args:
        result_files: List of JSON result files
        output_path: Path to save aggregated comparison
    
    Returns:
        Dict with comparison data
    """
    import pandas as pd
    
    experiments = {}
    
    for filepath in result_files:
        tracker = MetricsTracker.load(filepath)
        experiments[tracker.experiment_name] = tracker.get_aggregated()
    
    # Build comparison table
    rows = []
    for exp_name, metrics in experiments.items():
        row = {"experiment": exp_name}
        for metric_name, stats in metrics.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return experiments
