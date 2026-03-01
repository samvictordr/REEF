"""
Statistical tests for comparing models.

Provides:
- Paired t-test and Wilcoxon signed-rank test
- Bootstrap confidence intervals
- Multiple comparison correction (Bonferroni, FDR)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


def paired_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    test: str = "wilcoxon",
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Perform paired statistical test comparing two models.
    
    Args:
        scores_a: Scores from model A (per-fold)
        scores_b: Scores from model B (per-fold)
        test: 'wilcoxon' or 'ttest'
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Tuple of (statistic, p-value)
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")
    
    if len(scores_a) < 3:
        logger.warning("Very few samples for statistical test, results may be unreliable")
    
    if test == "wilcoxon":
        try:
            stat, p = stats.wilcoxon(
                scores_a - scores_b,
                alternative=alternative,
                zero_method="wilcox"
            )
        except ValueError:
            # All differences are zero
            stat, p = 0.0, 1.0
    elif test == "ttest":
        stat, p = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return float(stat), float(p)


def bootstrap_confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    statistic: str = "mean",
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        scores: Sample scores
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        statistic: 'mean' or 'median'
        random_state: Random seed
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    scores = np.asarray(scores)
    
    # Point estimate
    if statistic == "mean":
        point_estimate = np.mean(scores)
    else:
        point_estimate = np.median(scores)
    
    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        else:
            bootstrap_stats.append(np.median(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return float(point_estimate), float(ci_lower), float(ci_upper)


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Correct for multiple comparisons.
    
    Args:
        p_values: List of p-values
        method: 'bonferroni', 'holm', or 'fdr_bh'
        alpha: Significance level
    
    Returns:
        Tuple of (corrected_p_values, significant_flags)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    
    if method == "bonferroni":
        corrected = np.minimum(p_values * n, 1.0)
        significant = corrected < alpha
    
    elif method == "holm":
        # Holm-Bonferroni step-down
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros(n)
        
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = p_values[idx] * (n - i)
        
        corrected = np.minimum(corrected, 1.0)
        significant = corrected < alpha
    
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_values)
        ranks = np.arange(1, n + 1)
        
        corrected = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = p_values[idx] * n / (i + 1)
        
        # Ensure monotonicity
        corrected_sorted = corrected[sorted_idx]
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        corrected[sorted_idx] = corrected_sorted
        
        corrected = np.minimum(corrected, 1.0)
        significant = corrected < alpha
    
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return corrected.tolist(), significant.tolist()


def compare_models(
    results: Dict[str, List[float]],
    metric: str = "auroc",
    baseline: Optional[str] = None,
    test: str = "wilcoxon",
    correction: str = "bonferroni",
    alpha: float = 0.05,
) -> Dict:
    """
    Compare multiple models statistically.
    
    Args:
        results: Dict mapping model name to list of per-fold scores
        metric: Metric name (for logging)
        baseline: Baseline model name (compare all to this)
        test: Statistical test ('wilcoxon' or 'ttest')
        correction: Multiple comparison correction method
        alpha: Significance level
    
    Returns:
        Dict with comparison results
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    
    if baseline is None:
        baseline = model_names[0]
    
    if baseline not in results:
        raise ValueError(f"Baseline '{baseline}' not in results")
    
    baseline_scores = np.array(results[baseline])
    
    # Compute statistics for each model
    model_stats = {}
    comparisons = []
    
    for name, scores in results.items():
        scores = np.array(scores)
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(scores)
        
        model_stats[name] = {
            "mean": mean,
            "std": float(np.std(scores)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
        
        if name != baseline:
            stat, p = paired_test(scores, baseline_scores, test=test)
            comparisons.append({
                "model": name,
                "vs": baseline,
                "statistic": stat,
                "p_value": p,
                "mean_diff": float(np.mean(scores - baseline_scores)),
            })
    
    # Correct for multiple comparisons
    if comparisons:
        p_values = [c["p_value"] for c in comparisons]
        corrected_p, significant = multiple_comparison_correction(
            p_values, method=correction, alpha=alpha
        )
        
        for i, comp in enumerate(comparisons):
            comp["p_corrected"] = corrected_p[i]
            comp["significant"] = significant[i]
    
    return {
        "metric": metric,
        "baseline": baseline,
        "model_stats": model_stats,
        "comparisons": comparisons,
        "test": test,
        "correction": correction,
        "alpha": alpha,
    }


def format_comparison_table(comparison_results: Dict) -> str:
    """Format comparison results as a table string."""
    lines = [
        f"Metric: {comparison_results['metric']}",
        f"Baseline: {comparison_results['baseline']}",
        f"Test: {comparison_results['test']}, Correction: {comparison_results['correction']}",
        "",
        "Model".ljust(25) + "Mean ± Std".ljust(20) + "Diff vs Baseline".ljust(18) + "p-value".ljust(12) + "Sig?",
        "-" * 85,
    ]
    
    baseline = comparison_results["baseline"]
    
    # Baseline first
    stats = comparison_results["model_stats"][baseline]
    lines.append(
        f"{baseline.ljust(25)}"
        f"{stats['mean']:.4f} ± {stats['std']:.4f}".ljust(20)
        + "-".ljust(18) + "-".ljust(12) + "-"
    )
    
    # Other models
    for comp in comparison_results["comparisons"]:
        name = comp["model"]
        stats = comparison_results["model_stats"][name]
        sig_marker = "***" if comp["significant"] else ""
        
        lines.append(
            f"{name.ljust(25)}"
            f"{stats['mean']:.4f} ± {stats['std']:.4f}".ljust(20)
            + f"{comp['mean_diff']:+.4f}".ljust(18)
            + f"{comp['p_corrected']:.4f}".ljust(12)
            + sig_marker
        )
    
    return "\n".join(lines)
