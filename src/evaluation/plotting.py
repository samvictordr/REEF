"""
Publication-ready plotting functions for coral multi-omics experiments.

Provides:
- ROC curve plots
- Performance bar plots with error bars
- Confusion matrices
- Training curves
- Omics importance plots
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# Color palette for experiments
EXPERIMENT_COLORS = {
    "centralized": "#1f77b4",
    "vfl": "#ff7f0e",
    "mosaic_vfl": "#2ca02c",
    "complete_case": "#d62728",
    "imputation": "#9467bd",
    "grouped": "#8c564b",
    "ungrouped": "#e377c2",
    "transcriptomics": "#17becf",
    "metabolomics": "#bcbd22",
    "proteomics": "#7f7f7f",
    "microbiome": "#ff9896",
}


def plot_roc_curves(
    results: Dict[str, Dict],
    title: str = "ROC Curves Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: Dict mapping model name to dict with 'fpr', 'tpr', 'auroc'
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, data in results.items():
        fpr = data.get("fpr", [0, 1])
        tpr = data.get("tpr", [0, 1])
        auroc = data.get("auroc", 0.5)
        
        color = EXPERIMENT_COLORS.get(name.lower(), None)
        ax.plot(
            fpr, tpr,
            label=f"{name} (AUROC = {auroc:.3f})",
            color=color,
            linewidth=2,
        )
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved ROC curves to {output_path}")
    
    return fig


def plot_performance_barplot(
    results: Dict[str, Dict],
    metrics: List[str] = ["auroc", "f1", "accuracy"],
    title: str = "Model Performance Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot grouped bar chart comparing model performance.
    
    Args:
        results: Dict mapping model name to dict with metric values
                 Each metric should have 'mean' and 'std' keys
        metrics: List of metrics to plot
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_models
    
    for i, name in enumerate(model_names):
        means = []
        stds = []
        
        for metric in metrics:
            if metric in results[name]:
                data = results[name][metric]
                if isinstance(data, dict):
                    means.append(data.get("mean", data.get("value", 0)))
                    stds.append(data.get("std", 0))
                else:
                    means.append(data)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - n_models / 2 + 0.5) * bar_width
        color = EXPERIMENT_COLORS.get(name.lower(), None)
        
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            label=name,
            color=color,
            capsize=3,
            alpha=0.85,
        )
    
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved barplot to {output_path}")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str] = ["Ambient", "Stressed"],
    title: str = "Confusion Matrix",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (6, 5),
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        classes: Class labels
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by row (recall)
    
    Returns:
        matplotlib Figure
    """
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = cm_norm
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    
    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved confusion matrix to {output_path}")
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "AUROC",
    title: str = "Training Curves",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot training and validation loss/metric curves.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_metrics: Training metric per epoch
        val_metrics: Validation metric per epoch
        metric_name: Name of the metric
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax = axes[0]
    ax.plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, "r-", label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metric plot
    ax = axes[1]
    if train_metrics:
        ax.plot(epochs, train_metrics, "b-", label="Train", linewidth=2)
    if val_metrics:
        ax.plot(epochs, val_metrics, "r-", label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved training curves to {output_path}")
    
    return fig


def plot_omics_importance(
    importance: Dict[str, float],
    title: str = "Omics Block Importance",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot importance/contribution of each omics block.
    
    Args:
        importance: Dict mapping omics name to importance score
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    names = list(importance.keys())
    scores = list(importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(scores)[::-1]
    names = [names[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    
    colors = [EXPERIMENT_COLORS.get(n.lower(), "#333333") for n in names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    
    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved importance plot to {output_path}")
    
    return fig


def plot_experiment_comparison(
    experiments: Dict[str, Dict[str, float]],
    metric: str = "auroc",
    title: str = "Experiment Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot comparison across multiple experiments.
    
    Args:
        experiments: Dict mapping experiment name to metrics dict
        metric: Metric to compare
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    exp_names = list(experiments.keys())
    values = []
    errors = []
    
    for name in exp_names:
        data = experiments[name].get(metric, {})
        if isinstance(data, dict):
            values.append(data.get("mean", 0))
            errors.append(data.get("std", 0))
        else:
            values.append(data)
            errors.append(0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [EXPERIMENT_COLORS.get(n.lower().split("_")[0], "#333333") for n in exp_names]
    x = range(len(exp_names))
    
    ax.bar(x, values, yerr=errors, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved comparison plot to {output_path}")
    
    return fig


def plot_mosaic_vs_complete(
    complete_metrics: Dict[str, float],
    mosaic_metrics: Dict[str, float],
    n_complete: int,
    n_mosaic: int,
    title: str = "Mosaic vs Complete-Case Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot comparison between mosaic and complete-case approaches.
    
    Shows both performance metrics and sample sizes.
    
    Args:
        complete_metrics: Metrics for complete-case analysis
        mosaic_metrics: Metrics for mosaic analysis
        n_complete: Number of complete samples
        n_mosaic: Number of mosaic samples
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Sample size comparison
    ax = axes[0]
    sizes = [n_complete, n_mosaic]
    labels = [f"Complete\n(n={n_complete})", f"Mosaic\n(n={n_mosaic})"]
    colors = [EXPERIMENT_COLORS["complete_case"], EXPERIMENT_COLORS["mosaic_vfl"]]
    
    ax.bar(labels, sizes, color=colors, alpha=0.85)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Utilization")
    ax.grid(True, axis="y", alpha=0.3)
    
    # Performance comparison
    ax = axes[1]
    metrics = list(complete_metrics.keys())
    x = np.arange(len(metrics))
    bar_width = 0.35
    
    complete_vals = [complete_metrics[m] for m in metrics]
    mosaic_vals = [mosaic_metrics[m] for m in metrics]
    
    ax.bar(x - bar_width/2, complete_vals, bar_width, 
           label="Complete-case", color=colors[0], alpha=0.85)
    ax.bar(x + bar_width/2, mosaic_vals, bar_width,
           label="Mosaic", color=colors[1], alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved mosaic comparison to {output_path}")
    
    return fig


def create_paper_figure_1(
    roc_data: Dict,
    performance_data: Dict,
    output_path: Union[str, Path],
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Create Figure 1 for paper: VFL vs Centralized comparison.
    
    Combines ROC curves and performance barplot in a single figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel A: ROC curves
    ax = axes[0]
    for name, data in roc_data.items():
        fpr = data.get("fpr", [0, 1])
        tpr = data.get("tpr", [0, 1])
        auroc = data.get("auroc", 0.5)
        color = EXPERIMENT_COLORS.get(name.lower(), None)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auroc:.3f})", color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A) ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Panel B: Performance barplot
    ax = axes[1]
    metrics = ["auroc", "f1", "accuracy"]
    model_names = list(performance_data.keys())
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(model_names)
    
    for i, name in enumerate(model_names):
        means = [performance_data[name][m]["mean"] for m in metrics]
        stds = [performance_data[name][m]["std"] for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * bar_width
        color = EXPERIMENT_COLORS.get(name.lower(), None)
        ax.bar(x + offset, means, bar_width, yerr=stds, label=name, 
               color=color, capsize=3, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("B) Performance Metrics")
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path)
    logger.info(f"Saved Figure 1 to {output_path}")
    
    return fig
