# Evaluation module
"""Metrics, statistics, and plotting for coral multi-omics experiments."""

from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    MetricsTracker,
)
from .statistics import (
    paired_test,
    bootstrap_confidence_interval,
    multiple_comparison_correction,
    compare_models,
    format_comparison_table,
)
from .plotting import (
    plot_roc_curves,
    plot_performance_barplot,
    plot_confusion_matrix,
    plot_training_curves,
    plot_omics_importance,
    plot_experiment_comparison,
    plot_mosaic_vs_complete,
    create_paper_figure_1,
)

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "MetricsTracker",
    "paired_test",
    "bootstrap_confidence_interval",
    "multiple_comparison_correction",
    "compare_models",
    "format_comparison_table",
    "plot_roc_curves",
    "plot_performance_barplot",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_omics_importance",
    "plot_experiment_comparison",
    "plot_mosaic_vs_complete",
    "create_paper_figure_1",
]
