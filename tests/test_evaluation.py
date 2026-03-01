"""
Tests for evaluation metrics and statistics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from src.evaluation.statistics import (
    paired_ttest,
    wilcoxon_test,
    bootstrap_confidence_interval,
)


class TestClassificationMetrics:
    """Tests for classification metrics."""
    
    def test_perfect_classification(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        assert metrics.accuracy == 1.0
        assert metrics.auroc == 1.0
        assert metrics.f1 == 1.0
    
    def test_random_classification(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        y_prob = np.random.rand(n)
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        # Random should be around 0.5
        assert 0.3 <= metrics.auroc <= 0.7
        assert 0.3 <= metrics.accuracy <= 0.7
    
    def test_imbalanced_classification(self):
        """Test metrics with imbalanced classes."""
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.zeros(100)  # Always predict 0
        y_prob = np.ones(100) * 0.1
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        # Accuracy high due to class imbalance
        assert metrics.accuracy == 0.9
        # But F1 should be 0 for positive class
        assert metrics.sensitivity == 0.0


class TestRegressionMetrics:
    """Tests for regression metrics."""
    
    def test_perfect_regression(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert metrics.mse == 0.0
        assert metrics.mae == 0.0
        assert metrics.r2 == 1.0
    
    def test_random_regression(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randn(n) * 5
        y_pred = np.random.randn(n) * 5
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # R2 should be low for random
        assert metrics.r2 < 0.3
        # MSE and MAE should be non-zero
        assert metrics.mse > 0
        assert metrics.mae > 0


class TestStatisticalTests:
    """Tests for statistical comparison tests."""
    
    def test_paired_ttest_same(self):
        """Test paired t-test with identical samples."""
        a = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        b = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        
        result = paired_ttest(a, b)
        
        assert result["p_value"] == 1.0
        assert result["t_statistic"] == 0.0
    
    def test_paired_ttest_different(self):
        """Test paired t-test with different samples."""
        a = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        b = np.array([0.7, 0.72, 0.69, 0.71, 0.73])
        
        result = paired_ttest(a, b)
        
        # Should be significant
        assert result["p_value"] < 0.05
        assert result["significant"]
    
    def test_wilcoxon_different(self):
        """Test Wilcoxon test with different samples."""
        a = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.84])
        b = np.array([0.7, 0.72, 0.69, 0.71, 0.73, 0.68, 0.74])
        
        result = wilcoxon_test(a, b)
        
        # Should be significant
        assert result["p_value"] < 0.05
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval."""
        data = np.random.randn(100)
        
        ci = bootstrap_confidence_interval(data, n_iterations=1000)
        
        assert "lower" in ci
        assert "upper" in ci
        assert "mean" in ci
        assert ci["lower"] < ci["mean"] < ci["upper"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
