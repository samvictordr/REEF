"""
Domain Saliency Helper Functions for XAI-based Missing Omic Inference.

Provides utilities for computing domain saliency values and aggregating them
into fixed-length latent representations suitable for VFL fusion.

Dataset: Williams et al. (2021) - Montipora capitata thermal stress
DOI: 10.7717/peerj.12335
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import shap
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DomainExplainerWrapper:
    """
    Wrapper around domain saliency explainers with serialization and reproducibility.
    
    Supports both tree-based (LightGBM, XGBoost) and deep learning models.
    """
    
    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        explainer_type: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize domain saliency explainer wrapper.
        
        Args:
            model: Trained model (sklearn-compatible or PyTorch)
            background_data: Background dataset for saliency computation
            explainer_type: 'tree', 'kernel', 'deep', or 'auto'
            seed: Random seed for reproducibility
        """
        self.model = model
        self.background_data = background_data
        self.explainer_type = explainer_type
        self.seed = seed
        self.explainer = None
        
        np.random.seed(seed)
        
        self._create_explainer()
    
    def _create_explainer(self):
        """Create appropriate saliency explainer based on model type."""
        if self.explainer_type == "auto":
            # Auto-detect model type
            model_type = type(self.model).__name__.lower()
            
            if any(t in model_type for t in ['lightgbm', 'lgbm', 'xgb', 'randomforest', 'gradientboosting']):
                self.explainer_type = "tree"
            elif isinstance(self.model, nn.Module):
                self.explainer_type = "deep"
            else:
                self.explainer_type = "kernel"
        
        logger.info(f"Creating {self.explainer_type} saliency explainer")
        
        if self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == "deep":
            # For PyTorch models
            self.explainer = shap.DeepExplainer(
                self.model, 
                torch.tensor(self.background_data, dtype=torch.float32)
            )
        else:
            # Kernel explainer for any model
            self.explainer = shap.KernelExplainer(
                self.model.predict if hasattr(self.model, 'predict') else self.model,
                shap.sample(self.background_data, min(100, len(self.background_data)))
            )
    
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute saliency values for given samples.
        
        Args:
            X: Input samples (n_samples, n_features)
            
        Returns:
            Saliency values array (n_samples, n_features) or (n_samples, n_features, n_outputs)
        """
        if self.explainer_type == "deep":
            X_tensor = torch.tensor(X, dtype=torch.float32)
            shap_values = self.explainer.shap_values(X_tensor)
        else:
            shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            # Binary classification: take positive class
            shap_values = shap_values[1] if len(shap_values) == 2 else np.stack(shap_values, axis=-1)
        
        return np.array(shap_values)
    
    def save(self, path: Path):
        """Save explainer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'explainer_type': self.explainer_type,
            'seed': self.seed,
            'background_shape': list(self.background_data.shape),
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(path.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save explainer
        with open(path, 'wb') as f:
            pickle.dump({
                'explainer': self.explainer,
                'background_data': self.background_data,
            }, f)
        
        logger.info(f"Saved explainer to {path}")
    
    @classmethod
    def load(cls, path: Path, model: Any) -> "DomainExplainerWrapper":
        """Load explainer from disk."""
        path = Path(path)
        
        with open(path.with_suffix('.meta.json')) as f:
            metadata = json.load(f)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        wrapper = cls.__new__(cls)
        wrapper.model = model
        wrapper.background_data = data['background_data']
        wrapper.explainer_type = metadata['explainer_type']
        wrapper.seed = metadata['seed']
        wrapper.explainer = data['explainer']
        
        logger.info(f"Loaded explainer from {path}")
        return wrapper


def compute_domain_saliency_values(
    model: Any,
    X_explain: np.ndarray,
    X_background: np.ndarray,
    explainer_type: str = "auto",
    seed: int = 42,
) -> Tuple[np.ndarray, DomainExplainerWrapper]:
    """
    Compute saliency values for samples using specified model.
    
    Args:
        model: Trained model
        X_explain: Samples to explain
        X_background: Background dataset
        explainer_type: Type of saliency explainer
        seed: Random seed
        
    Returns:
        Tuple of (shap_values, explainer_wrapper)
    """
    wrapper = DomainExplainerWrapper(
        model=model,
        background_data=X_background,
        explainer_type=explainer_type,
        seed=seed,
    )
    
    shap_values = wrapper.explain(X_explain)
    
    return shap_values, wrapper


def aggregate_saliency_to_latent(
    shap_values: np.ndarray,
    method: str = "mean_abs",
    latent_dim: int = 64,
    feature_groups: Optional[Dict[str, List[int]]] = None,
) -> np.ndarray:
    """
    Aggregate SHAP values into fixed-length latent representation.
    
    This creates a proxy representation for a missing omic block by
    summarizing the SHAP contributions from available omics.
    
    Args:
        shap_values: SHAP values (n_samples, n_features)
        method: Aggregation method - 'mean_abs', 'sum', 'pca', 'bins'
        latent_dim: Target dimension for latent representation
        feature_groups: Optional grouping of features for structured aggregation
        
    Returns:
        Latent representation (n_samples, latent_dim)
    """
    n_samples, n_features = shap_values.shape[:2]
    
    if method == "mean_abs":
        # Split features into bins and compute mean absolute SHAP per bin
        bin_size = max(1, n_features // latent_dim)
        latent = np.zeros((n_samples, latent_dim))
        
        for i in range(latent_dim):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, n_features)
            if start_idx < n_features:
                latent[:, i] = np.mean(np.abs(shap_values[:, start_idx:end_idx]), axis=1)
    
    elif method == "sum":
        # Sum SHAP values in bins
        bin_size = max(1, n_features // latent_dim)
        latent = np.zeros((n_samples, latent_dim))
        
        for i in range(latent_dim):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, n_features)
            if start_idx < n_features:
                latent[:, i] = np.sum(shap_values[:, start_idx:end_idx], axis=1)
    
    elif method == "pca":
        # PCA projection of SHAP values
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(latent_dim, n_features, n_samples))
        latent_raw = pca.fit_transform(shap_values)
        
        # Pad if needed
        if latent_raw.shape[1] < latent_dim:
            latent = np.zeros((n_samples, latent_dim))
            latent[:, :latent_raw.shape[1]] = latent_raw
        else:
            latent = latent_raw[:, :latent_dim]
    
    elif method == "bins":
        # Quantile-based binning
        latent = np.zeros((n_samples, latent_dim))
        
        for i in range(n_samples):
            sample_shap = shap_values[i]
            # Sort and bin
            sorted_shap = np.sort(sample_shap)
            bin_edges = np.linspace(0, len(sorted_shap), latent_dim + 1, dtype=int)
            
            for j in range(latent_dim):
                latent[i, j] = np.mean(sorted_shap[bin_edges[j]:bin_edges[j+1]])
    
    elif method == "grouped" and feature_groups is not None:
        # Group-wise aggregation
        latent_parts = []
        
        for group_name, indices in feature_groups.items():
            group_shap = shap_values[:, indices]
            group_dim = max(1, latent_dim // len(feature_groups))
            
            # Mean absolute within group
            if len(indices) > group_dim:
                bin_size = len(indices) // group_dim
                group_latent = np.zeros((n_samples, group_dim))
                for i in range(group_dim):
                    start = i * bin_size
                    end = min((i + 1) * bin_size, len(indices))
                    group_latent[:, i] = np.mean(np.abs(group_shap[:, start:end]), axis=1)
            else:
                group_latent = np.mean(np.abs(group_shap), axis=1, keepdims=True)
            
            latent_parts.append(group_latent)
        
        latent = np.concatenate(latent_parts, axis=1)
        
        # Pad or truncate to exact latent_dim
        if latent.shape[1] < latent_dim:
            padded = np.zeros((n_samples, latent_dim))
            padded[:, :latent.shape[1]] = latent
            latent = padded
        else:
            latent = latent[:, :latent_dim]
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return latent.astype(np.float32)


def compute_inference_fidelity(
    original_latent: np.ndarray,
    inferred_latent: np.ndarray,
) -> Dict[str, float]:
    """
    Compute fidelity metrics between original and inferred latent representations.
    
    Args:
        original_latent: Ground truth latent vectors
        inferred_latent: Inferred latent vectors
        
    Returns:
        Dictionary of fidelity metrics
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metrics = {}
    
    # Overall MSE and MAE
    metrics['mse'] = float(mean_squared_error(original_latent, inferred_latent))
    metrics['mae'] = float(mean_absolute_error(original_latent, inferred_latent))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    # Per-sample correlations
    pearson_corrs = []
    spearman_corrs = []
    
    for i in range(len(original_latent)):
        if np.std(original_latent[i]) > 0 and np.std(inferred_latent[i]) > 0:
            r_p, _ = pearsonr(original_latent[i], inferred_latent[i])
            r_s, _ = spearmanr(original_latent[i], inferred_latent[i])
            pearson_corrs.append(r_p)
            spearman_corrs.append(r_s)
    
    metrics['pearson_r_mean'] = float(np.mean(pearson_corrs)) if pearson_corrs else 0.0
    metrics['pearson_r_std'] = float(np.std(pearson_corrs)) if pearson_corrs else 0.0
    metrics['spearman_rho_mean'] = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0
    metrics['spearman_rho_std'] = float(np.std(spearman_corrs)) if spearman_corrs else 0.0
    
    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sims = [cosine_similarity([original_latent[i]], [inferred_latent[i]])[0, 0] 
                for i in range(len(original_latent))]
    metrics['cosine_sim_mean'] = float(np.mean(cos_sims))
    metrics['cosine_sim_std'] = float(np.std(cos_sims))
    
    return metrics
