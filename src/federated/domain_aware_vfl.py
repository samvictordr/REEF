#!/usr/bin/env python3
# REEF: Robust Expert Encoder Federation
"""
DomainAware Vertical Federated Learning for Coral Multi-Omics.

This module extends the base NVFlare VFL with domain-aware feature selection
to improve accuracy on small, high-dimensional multi-omic datasets.

Key Innovations:
1. Two-phase training: Warmup → Domain selection → Refinement
2. Domain-aware feature selection: Uses gradient saliency to identify stress-response features
3. Iterative refinement: Multiple domain selection passes for optimal feature sets
4. Omic-specific weighting: Different top-K per omic based on information content

Why Domain-Aware Selection Helps Multi-Omics:
- Small N, Large P: 13 samples vs 106K features → severe overfitting risk
- Domain selection identifies ~500 biologically relevant features per omic
- Reduces noise from irrelevant metabolites/genes
- Focuses model on stress-response signals (heat-shock proteins, etc.)
- Privacy-preserving: Only importance values shared, not raw data
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.nvflare_vfl_runner import (
    NVFlareVFLTrainer,
    VFLConfig,
    OmicSilo,
    OmicsEncoder,
    VFLFusionHead,
    load_coral_data,
    compute_domain_saliency,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DomainAwareConfig(VFLConfig):
    """Extended config for domain-aware training."""
    # Domain feature selection
    topk_features: int = 500  # Features to keep per omic
    domain_warmup_rounds: int = 20  # Initial training before domain selection
    domain_refinement_rounds: int = 2  # Number of domain refinement iterations
    
    # Domain-aware settings: feature selection ratios
    # These control how many features to retain per omic (higher = keep more)
    omic_topk_ratios: Dict[str, float] = field(default_factory=lambda: {
        "proteomics": 0.5,     # Keep 50% of selected features (smaller, denser)
        "metabolomics": 1.0,  # Keep 100% (medium size)
        "microbiome": 0.8,    # Keep 80% (sparse)
        "transcriptomics": 0.3,  # Keep 30% (very large, sparse)
    })
    
    # Biological importance weights for omic layers
    # Applied to EMBEDDINGS before fusion (not during feature selection)
    # Scales each omic's contribution to the fusion head and its gradient signal
    # Rationale based on coral stress biology:
    # - Transcriptomics (1.5×): Primary regulatory driver (mRNA reflects active gene expression)
    # - Proteomics (1.0×): Effector molecules (translation products, baseline reference)
    # - Metabolomics (0.8×): Functional readout (downstream of transcription/translation)
    # - Microbiome (0.5×): Environmental/symbiont confounder (indirect holobiont effects)
    omic_biological_weights: Dict[str, float] = field(default_factory=lambda: {
        "transcriptomics": 1.5,
        "proteomics": 1.0,
        "metabolomics": 0.8,
        "microbiome": 0.5,
    })
    
    # Feature selection method
    selection_method: str = "topk"  # "topk", "threshold", "percentile"
    importance_threshold: float = 0.001  # For threshold method
    importance_percentile: float = 95  # For percentile method
    
    # Regularization
    feature_sparsity_weight: float = 0.01  # L1 on unused features
    importance_smoothing: float = 0.1  # EMA for importance across iterations


# =============================================================================
# REEF Trainer
# =============================================================================

class REEF(NVFlareVFLTrainer):
    """
    Vertical Federated Learning with domain-aware feature selection.
    
    This trainer improves accuracy on small multi-omic datasets by:
    1. Training initial model to learn basic patterns
    2. Using gradient saliency to identify important features per omic
    3. Masking low-importance features to reduce noise
    4. Retraining with focused feature set
    
    The key insight is that with 13 samples and 106K features,
    the model will overfit to noise. Domain-aware selection helps identify the
    ~1-2% of features that carry actual biological signal.
    """
    
    def __init__(
        self,
        silos: Dict[str, OmicSilo],
        labels: np.ndarray,
        config: DomainAwareConfig,
    ):
        super().__init__(silos, labels, config)
        
        self.domain_config = config
        self.feature_masks: Dict[str, torch.Tensor] = {}
        self.domain_importance: Dict[str, np.ndarray] = {}
        self.importance_history: List[Dict[str, np.ndarray]] = []
        self.training_phase: str = "warmup"
        
        # Track per-omic statistics
        self.omic_stats = {
            name: {
                "n_features": silo.n_features,
                "selected_features": 0,
                "top_importance": 0.0,
                "mean_importance": 0.0,
            }
            for name, silo in silos.items()
        }
    
    def _compute_omic_topk(self, omic_name: str) -> int:
        """Compute top-K for a specific omic based on domain knowledge."""
        base_k = self.domain_config.topk_features
        ratio = self.domain_config.omic_topk_ratios.get(omic_name, 1.0)
        n_features = self.silos[omic_name].n_features
        
        # Compute target K
        target_k = int(base_k * ratio)
        
        # Ensure reasonable bounds
        min_k = min(50, n_features)  # At least 50 features
        max_k = min(n_features, 2000)  # Cap at 2000
        
        return max(min_k, min(target_k, max_k))
    
    def _compute_domain_saliency(
        self, 
        train_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature importance with smoothing across iterations.
        
        Uses gradient-based explainers (expected gradients) as primary method,
        with Integrated Gradients as fallback. Both methods provide interpretable
        feature attributions suitable for domain-aware feature selection.
        """
        logger.info("Computing feature importance via interpretable XAI methods...")
        
        try:
            import shap
            has_shap = True
        except ImportError:
            logger.warning("SHAP not installed, using Integrated Gradients")
            has_shap = False
        
        # Get embeddings for background
        embeddings = self.get_embeddings(train_indices, train_indices)
        
        new_importance = {}
        
        for omic_name in self.omic_names:
            logger.info(f"  Computing importance for {omic_name}...")
            
            silo = self.silos[omic_name]
            
            # Background data
            bg_indices = np.random.choice(
                train_indices, 
                size=min(50, len(train_indices)), 
                replace=False
            )
            bg_data = silo.scaler.transform(silo.features[bg_indices])
            bg_data = np.clip(bg_data, -5, 5)
            bg_data = np.nan_to_num(bg_data, nan=0.0)
            bg_tensor = torch.tensor(bg_data, dtype=torch.float32, device=self.device).clone()
            
            # Test data
            test_data = silo.scaler.transform(silo.features[train_indices])
            test_data = np.clip(test_data, -5, 5)
            test_data = np.nan_to_num(test_data, nan=0.0)
            test_tensor = torch.tensor(test_data, dtype=torch.float32, device=self.device).clone()
            
            importance = None
            method_used = None
            
            # Method 1: Try SHAP GradientExplainer
            if has_shap:
                try:
                    from src.federated.nvflare_vfl_runner import VFLDomainWrapper
                    wrapper = VFLDomainWrapper(self, omic_name, embeddings)
                    wrapper.eval()
                    
                    explainer = shap.GradientExplainer(wrapper, bg_tensor)
                    shap_values = explainer.shap_values(test_tensor)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    importance = np.abs(np.array(shap_values)).mean(axis=0)
                    method_used = "GradientExplainer (SHAP)"
                    logger.info(f"    {omic_name}: Using GradientExplainer")
                    
                except Exception as e:
                    logger.warning(f"    {omic_name}: GradientExplainer failed ({e})")
            
            # Method 2: Integrated Gradients
            if importance is None:
                try:
                    importance = self._compute_integrated_gradients_single(
                        omic_name, test_tensor, bg_tensor
                    )
                    method_used = "Integrated Gradients"
                    logger.info(f"    {omic_name}: Using Integrated Gradients")
                except Exception as e:
                    logger.warning(f"    {omic_name}: Integrated Gradients failed ({e})")
            
            # Method 3: Gradient saliency fallback
            if importance is None:
                importance = self._compute_gradient_importance_single(omic_name, train_indices)
                method_used = "Gradient Saliency"
                logger.info(f"    {omic_name}: Using gradient saliency fallback")
            
            # Normalize importance
            if importance.max() > 0:
                importance = importance / importance.max()
            
            new_importance[omic_name] = importance
        
        # Apply exponential moving average with previous importance
        if self.domain_importance and self.domain_config.importance_smoothing > 0:
            alpha = self.domain_config.importance_smoothing
            for name in new_importance:
                if name in self.domain_importance:
                    new_importance[name] = (
                        alpha * self.domain_importance[name] + 
                        (1 - alpha) * new_importance[name]
                    )
        
        self.importance_history.append(new_importance)
        self.domain_importance = new_importance
        
        return new_importance
    
    def _compute_integrated_gradients_single(
        self,
        omic_name: str,
        features: torch.Tensor,
        baseline: torch.Tensor,
        n_steps: int = 50,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients (Sundararajan et al., 2017) for a single omic.
        
        Integrated Gradients is an axiomatically-grounded attribution method
        that satisfies completeness, sensitivity, and implementation invariance.
        
        Args:
            omic_name: Name of the omic layer
            features: Input features tensor (n_samples, n_features)
            baseline: Baseline features tensor (n_background, n_features)
            n_steps: Number of interpolation steps
            
        Returns:
            Per-feature importance scores (n_features,)
        """
        silo = self.silos[omic_name]
        encoder = silo.encoder
        encoder.eval()
        
        # Use mean of background as baseline
        baseline_mean = baseline.mean(dim=0, keepdim=True)
        baseline_expanded = baseline_mean.expand(features.shape[0], -1)
        
        # Compute scaled inputs along interpolation path
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        # Accumulate gradients along path
        integrated_grads = torch.zeros_like(features)
        
        for alpha in alphas:
            interpolated = baseline_expanded + alpha * (features - baseline_expanded)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            output = encoder(interpolated)
            target = output.sum()
            target.backward()
            
            if interpolated.grad is not None:
                integrated_grads += interpolated.grad
        
        # Average gradients (Riemann sum approximation)
        integrated_grads = integrated_grads / n_steps
        
        # Scale by (input - baseline) for final attributions
        attributions = (features - baseline_expanded) * integrated_grads
        
        # Mean absolute attribution across samples
        importance = torch.abs(attributions).mean(dim=0).detach().cpu().numpy()
        
        return importance
    
    def _compute_gradient_importance_single(
        self,
        omic_name: str,
        train_indices: np.ndarray,
    ) -> np.ndarray:
        """Fallback: gradient-based feature importance for one omic."""
        silo = self.silos[omic_name]
        
        # Prepare data
        data = silo.scaler.transform(silo.features[train_indices])
        data = np.clip(data, -5, 5)
        data = np.nan_to_num(data, nan=0.0)
        X = torch.tensor(data, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Forward pass
        self.fusion_head.eval()
        silo.encoder.eval()
        
        emb = silo.encoder(X)
        
        # Use embedding norm as proxy for importance
        importance = torch.abs(X.grad).mean(dim=0).cpu().numpy() if X.grad is not None else np.ones(silo.n_features)
        
        # Alternative: use weight magnitudes
        first_layer = list(silo.encoder.net.children())[0]
        if hasattr(first_layer, 'weight'):
            weight_importance = torch.abs(first_layer.weight).mean(dim=0).cpu().detach().numpy()
            importance = weight_importance
        
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return importance
    
    def _compute_gradient_importance(
        self, 
        train_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Gradient-based importance as fallback."""
        importance = {}
        for omic_name in self.omic_names:
            importance[omic_name] = self._compute_gradient_importance_single(
                omic_name, train_indices
            )
        return importance
    
    def _create_feature_masks(self):
        """Create feature masks based on domain saliency.
        
        Feature selection is per-omic using importance scores and omic_topk_ratios.
        
        NOTE: Biological weights (omic_biological_weights) are applied at the
        EMBEDDING level in _client_forward(), not here. Within-omic ranking
        is invariant to uniform scaling (argsort(x*c) == argsort(x)), so
        bio weights must operate on the embeddings to have a real effect.
        """
        logger.info("Creating feature masks based on domain saliency...")
        
        for omic_name, importance in self.domain_importance.items():
            # Ensure importance is 1D
            importance = np.asarray(importance).flatten()
            n_features = len(importance)
            
            bio_weight = self.domain_config.omic_biological_weights.get(omic_name, 1.0)
            
            # Get omic-specific top-K
            if self.domain_config.selection_method == "topk":
                k = self._compute_omic_topk(omic_name)
                top_indices = np.argsort(importance)[-k:]
                
            elif self.domain_config.selection_method == "threshold":
                top_indices = np.where(importance >= self.domain_config.importance_threshold)[0]
                if len(top_indices) < 50:
                    top_indices = np.argsort(importance)[-50:]
                    
            elif self.domain_config.selection_method == "percentile":
                threshold = np.percentile(importance, self.domain_config.importance_percentile)
                top_indices = np.where(importance >= threshold)[0]
                if len(top_indices) < 50:
                    top_indices = np.argsort(importance)[-50:]
            
            # Create binary mask
            mask = torch.zeros(n_features, dtype=torch.float32)
            mask[top_indices] = 1.0
            self.feature_masks[omic_name] = mask
            
            # Update stats
            n_selected = len(top_indices)
            top_imp = float(importance[top_indices[-1]]) if n_selected > 0 else 0.0
            mean_imp = float(importance[top_indices].mean()) if n_selected > 0 else 0.0
            
            self.omic_stats[omic_name]["selected_features"] = n_selected
            self.omic_stats[omic_name]["top_importance"] = top_imp
            self.omic_stats[omic_name]["mean_importance"] = mean_imp
            
            logger.info(
                f"  {omic_name}: {n_selected}/{n_features} features "
                f"(top={top_imp:.4f}, mean={mean_imp:.4f}, "
                f"bio_weight={bio_weight:.2f}x applied at embedding level)"
            )
    
    def _client_forward(
        self, 
        features_dict: Dict[str, torch.Tensor],
        retain_grads: bool = True,
    ) -> List[torch.Tensor]:
        """
        Client forward pass with domain-aware embedding weighting.
        
        Applies omic_biological_weights to embeddings AFTER encoding,
        so each omic's contribution to the fusion head is scaled by its
        biological importance prior:
          transcriptomics x 1.5 (primary regulatory driver)
          proteomics      x 1.0 (effector molecules, baseline)
          metabolomics    x 0.8 (downstream functional readout)
          microbiome      x 0.5 (environmental confounder)
        
        This also scales the backward gradient proportionally, so
        higher-weighted omics receive stronger learning signal.
        """
        embeddings = []
        for name in self.omic_names:
            silo = self.silos[name]
            emb = silo.encoder(features_dict[name])
            
            # Domain-aware prior: scale embedding by biological importance weight
            bio_weight = self.domain_config.omic_biological_weights.get(name, 1.0)
            emb = emb * bio_weight
            
            if retain_grads and emb.requires_grad:
                emb.retain_grad()  # For backward pass
            embeddings.append(emb)
        return embeddings
    
    def _apply_feature_mask(
        self, 
        features_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply feature masks to input features."""
        masked = {}
        for name, features in features_dict.items():
            if name in self.feature_masks:
                mask = self.feature_masks[name].to(features.device)
                masked[name] = features * mask
            else:
                masked[name] = features
        return masked
    
    def _prepare_batch(
        self, 
        indices: np.ndarray,
        train_indices: np.ndarray,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare batch with optional feature masking."""
        features_dict, labels = super()._prepare_batch(indices, train_indices)
        
        # Apply masks during refinement phase
        if self.training_phase == "refinement" and self.feature_masks:
            features_dict = self._apply_feature_mask(features_dict)
        
        return features_dict, labels
    
    def train(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Multi-phase domain-aware training:
        
        Phase 1 (Warmup): Train on all features to learn initial patterns
        Phase 2 (Domain Selection): Compute importance, select top-K features
        Phase 3 (Refinement): Retrain with masked features
        
        Optional: Multiple refinement iterations with updated domain saliency
        """
        logger.info("=" * 70)
        logger.info("REEF Vertical Federated Learning")
        logger.info("=" * 70)
        logger.info(f"Config: topk={self.domain_config.topk_features}, "
                   f"warmup={self.domain_config.domain_warmup_rounds}, "
                   f"refinements={self.domain_config.domain_refinement_rounds}")
        
        full_history = {
            "phases": [],
            "domain_importance": [],
            "feature_stats": [],
        }
        
        # Store original config
        original_n_rounds = self.config.n_rounds
        total_rounds_used = 0
        
        # =====================================================================
        # Phase 1: Warmup Training
        # =====================================================================
        self.training_phase = "warmup"
        warmup_rounds = min(self.domain_config.domain_warmup_rounds, original_n_rounds // 2)
        
        logger.info(f"\n[Phase 1/3] Warmup training for {warmup_rounds} rounds...")
        self.config.n_rounds = warmup_rounds
        self.patience_counter = 0
        self.best_val_auroc = 0.0
        
        warmup_history = super().train(train_indices, val_indices)
        total_rounds_used += len(warmup_history["train_loss"])
        
        full_history["phases"].append({
            "name": "warmup",
            "rounds": len(warmup_history["train_loss"]),
            "best_val_auroc": self.best_val_auroc,
            "train_loss": warmup_history["train_loss"],
        })
        
        # =====================================================================
        # Phase 2: Domain Feature Selection
        # =====================================================================
        logger.info(f"\n[Phase 2/3] Computing domain saliency...")
        self._compute_domain_importance(train_indices)
        self._create_feature_masks()
        
        full_history["domain_importance"].append({
            name: imp.tolist() for name, imp in self.domain_importance.items()
        })
        full_history["feature_stats"].append(dict(self.omic_stats))
        
        # =====================================================================
        # Phase 3: Refinement Training with Feature Masking
        # =====================================================================
        self.training_phase = "refinement"
        
        remaining_rounds = original_n_rounds - total_rounds_used
        rounds_per_refinement = remaining_rounds // max(1, self.domain_config.domain_refinement_rounds)
        
        for refine_iter in range(self.domain_config.domain_refinement_rounds):
            logger.info(f"\n[Phase 3.{refine_iter+1}] Refinement iteration {refine_iter+1}...")
            
            self.config.n_rounds = rounds_per_refinement
            self.patience_counter = 0  # Reset early stopping
            
            refinement_history = super().train(train_indices, val_indices)
            total_rounds_used += len(refinement_history["train_loss"])
            
            full_history["phases"].append({
                "name": f"refinement_{refine_iter+1}",
                "rounds": len(refinement_history["train_loss"]),
                "best_val_auroc": self.best_val_auroc,
                "train_loss": refinement_history["train_loss"],
            })
            
            # Update domain saliency if not last iteration
            if refine_iter < self.domain_config.domain_refinement_rounds - 1:
                logger.info("  Updating domain saliency...")
                self._compute_domain_importance(train_indices)
                self._create_feature_masks()
                
                full_history["domain_importance"].append({
                    name: imp.tolist() for name, imp in self.domain_importance.items()
                })
                full_history["feature_stats"].append(dict(self.omic_stats))
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("REEF Training Complete")
        logger.info("=" * 70)
        logger.info(f"Total rounds: {total_rounds_used}")
        logger.info(f"Best validation AUROC: {self.best_val_auroc:.4f}")
        
        for name, stats in self.omic_stats.items():
            logger.info(
                f"  {name}: {stats['selected_features']}/{stats['n_features']} features "
                f"({100*stats['selected_features']/stats['n_features']:.1f}%)"
            )
        
        # Restore config
        self.config.n_rounds = original_n_rounds
        
        # Add final stats
        full_history["total_rounds"] = total_rounds_used
        full_history["best_val_auroc"] = self.best_val_auroc
        full_history["omic_stats"] = self.omic_stats
        full_history["feature_masks"] = {
            name: mask.numpy().tolist() for name, mask in self.feature_masks.items()
        }
        
        return full_history
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance for analysis."""
        summary = {}
        
        for omic_name in self.omic_names:
            if omic_name not in self.domain_importance:
                continue
                
            importance = self.domain_importance[omic_name]
            mask = self.feature_masks.get(omic_name)
            
            # Get top features
            top_k = min(20, len(importance))
            top_indices = np.argsort(importance)[-top_k:][::-1]
            
            summary[omic_name] = {
                "n_total": len(importance),
                "n_selected": int(mask.sum()) if mask is not None else len(importance),
                "selection_ratio": float(mask.sum() / len(mask)) if mask is not None else 1.0,
                "top_20_indices": top_indices.tolist(),
                "top_20_importance": importance[top_indices].tolist(),
                "mean_importance": float(importance.mean()),
                "max_importance": float(importance.max()),
                "sparsity": float((importance < 0.01).mean()),
            }
        
        return summary


# =============================================================================
# Ablation Study Runner
# =============================================================================

def run_ablation_study(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    topk_values: List[int] = [100, 250, 500, 1000, 2000],
    warmup_values: List[int] = [10, 20, 30],
) -> Dict[str, Any]:
    """
    Run ablation study on domain-aware training hyperparameters.
    
    Tests:
    1. Different top-K feature counts
    2. Different warmup round counts
    3. With/without iterative refinement
    """
    logger.info("=" * 70)
    logger.info("REEF VFL Ablation Study")
    logger.info("=" * 70)
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data once
    silos, labels, splits = load_coral_data(data_dir, seed=seed)
    
    results = {
        "seed": seed,
        "topk_ablation": [],
        "warmup_ablation": [],
        "refinement_ablation": [],
    }
    
    # =========================================================================
    # Ablation 1: Top-K Features
    # =========================================================================
    logger.info("\n[Ablation 1] Testing top-K feature values...")
    
    for topk in topk_values:
        logger.info(f"\n  Testing topk={topk}...")
        
        config = DomainAwareConfig(
            n_rounds=60,
            topk_features=topk,
            domain_warmup_rounds=20,
            domain_refinement_rounds=1,
            lr_client=5e-5,
            lr_server=5e-4,
            seed=seed,
        )
        
        trainer = REEF(silos, labels, config)
        history = trainer.train(splits["train"], splits["val"])
        test_metrics = trainer.evaluate(splits["test"], splits["train"])
        
        results["topk_ablation"].append({
            "topk": topk,
            "test_auroc": test_metrics["auroc"],
            "test_accuracy": test_metrics["accuracy"],
            "best_val_auroc": trainer.best_val_auroc,
            "total_selected_features": sum(
                s["selected_features"] for s in trainer.omic_stats.values()
            ),
        })
        
        logger.info(f"    AUROC: {test_metrics['auroc']:.4f}")
    
    # =========================================================================
    # Ablation 2: Warmup Rounds
    # =========================================================================
    logger.info("\n[Ablation 2] Testing warmup round values...")
    
    for warmup in warmup_values:
        logger.info(f"\n  Testing warmup={warmup}...")
        
        config = DomainAwareConfig(
            n_rounds=60,
            topk_features=500,
            domain_warmup_rounds=warmup,
            domain_refinement_rounds=1,
            lr_client=5e-5,
            lr_server=5e-4,
            seed=seed,
        )
        
        trainer = REEF(silos, labels, config)
        history = trainer.train(splits["train"], splits["val"])
        test_metrics = trainer.evaluate(splits["test"], splits["train"])
        
        results["warmup_ablation"].append({
            "warmup_rounds": warmup,
            "test_auroc": test_metrics["auroc"],
            "test_accuracy": test_metrics["accuracy"],
            "best_val_auroc": trainer.best_val_auroc,
        })
        
        logger.info(f"    AUROC: {test_metrics['auroc']:.4f}")
    
    # =========================================================================
    # Ablation 3: Refinement Iterations
    # =========================================================================
    logger.info("\n[Ablation 3] Testing refinement iterations...")
    
    for n_refine in [0, 1, 2, 3]:
        logger.info(f"\n  Testing refinements={n_refine}...")
        
        config = DomainAwareConfig(
            n_rounds=60,
            topk_features=500,
            domain_warmup_rounds=20,
            domain_refinement_rounds=max(1, n_refine),  # At least 1 for training
            lr_client=5e-5,
            lr_server=5e-4,
            seed=seed,
        )
        
        trainer = REEF(silos, labels, config)
        
        if n_refine == 0:
            # No refinement - just warmup
            trainer.training_phase = "warmup"
            trainer.config.n_rounds = 60
            history = NVFlareVFLTrainer.train(trainer, splits["train"], splits["val"])
            test_metrics = trainer.evaluate(splits["test"], splits["train"])
        else:
            history = trainer.train(splits["train"], splits["val"])
            test_metrics = trainer.evaluate(splits["test"], splits["train"])
        
        results["refinement_ablation"].append({
            "n_refinements": n_refine,
            "test_auroc": test_metrics["auroc"],
            "test_accuracy": test_metrics["accuracy"],
            "best_val_auroc": trainer.best_val_auroc,
        })
        
        logger.info(f"    AUROC: {test_metrics['auroc']:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nAblation results saved to {results_path}")
    
    return results


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="REEF VFL for Coral Multi-Omics"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/domain_vfl")
    parser.add_argument("--n-rounds", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Paths
    project_root = PROJECT_ROOT
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.ablation:
        # Run ablation study
        results = run_ablation_study(data_dir, output_dir, seed=args.seed)
        return results
    
    # Load data
    silos, labels, splits = load_coral_data(data_dir, seed=args.seed)
    
    # Config
    config = DomainAwareConfig(
        n_rounds=args.n_rounds,
        topk_features=args.topk,
        domain_warmup_rounds=args.warmup_rounds,
        domain_refinement_rounds=args.refinements,
        lr_client=5e-5,
        lr_server=5e-4,
        seed=args.seed,
    )
    
    # Train
    trainer = REEF(silos, labels, config)
    history = trainer.train(splits["train"], splits["val"])
    
    # Evaluate
    test_metrics = trainer.evaluate(splits["test"], splits["train"])
    logger.info(f"\nTest AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Get feature importance summary
    importance_summary = trainer.get_feature_importance_summary()
    
    # Save results
    checkpoint_path = output_dir / f"domain_vfl_seed{args.seed}.pt"
    trainer.save_checkpoint(checkpoint_path)
    
    metrics = {
        "test_auroc": test_metrics["auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "best_val_auroc": trainer.best_val_auroc,
        "config": {
            "n_rounds": args.n_rounds,
            "topk_features": args.topk,
            "warmup_rounds": args.warmup_rounds,
            "refinements": args.refinements,
        },
        "omic_stats": trainer.omic_stats,
        "importance_summary": importance_summary,
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
