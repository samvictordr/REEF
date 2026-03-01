"""
Experiment H2: Use Groups with XAI-Inferred Missing Omics

This is the CORE experiment for the coral multi-omics VFL project.

When an omic block (e.g., proteomics) is missing for a sample, we compute
an XAI/gradient saliency-based inferred latent representation from the available omics
and feed that to the VFL fusion head in place of the missing block.

Strategies compared:
1. Complete-case: Only use samples with all omics (baseline)
2. Zero imputation: Replace missing omics with zeros
3. Mean imputation: Replace missing with global mean
4. Mosaic (mask): Use per-sample availability masks
5. XAI-inferred: Use gradient saliency-based proxy for missing omics (main contribution)

Dataset: Williams et al. (2021) - Multi-omic characterization of thermal stress
in Montipora capitata. DOI: 10.7717/peerj.12335

Usage:
    # Toy mode for quick validation
    python -m src.experiments.experiment_h2_use_groups --config configs/h2_config.yaml --toy
    
    # Full experiment
    python -m src.experiments.experiment_h2_use_groups --config configs/h2_config.yaml
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .base_experiment import BaseExperiment
from ..data import (
    CoralMultiOmicsLoader,
    MosaicMultiOmicsDataset,
    create_cv_splits,
)
from ..models import (
    LateFusionModel,
    MosaicVFLModel,
)
from ..xai import (
    MissingOmicInferrer,
    train_inference_model,
    compute_inference_fidelity,
)
from ..evaluation import (
    MetricsTracker,
    compute_classification_metrics,
    compare_models,
    format_comparison_table,
)

logger = logging.getLogger(__name__)


class UseGroupsExperiment(BaseExperiment):
    """
    H2 Experiment: Use Groups with XAI Inference for Missing Omics.
    
    This experiment tests whether gradient saliency-based inference of missing omic
    representations improves upon simple imputation strategies in VFL.
    """
    
    STRATEGIES = [
        "complete_case",
        "zero_imputation", 
        "mean_imputation",
        "mosaic_mask",
        "xai_inferred",
    ]
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path = Path("outputs/h2"),
        device: Optional[str] = None,
        toy_mode: bool = False,
    ):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
            output_dir: Output directory
            device: Device to use (cuda/cpu)
            toy_mode: Run with minimal data for testing
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.toy_mode = toy_mode
        
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.seed = config.get("seed", 42)
        self._set_seeds()
        
        # Data containers
        self.data = None
        self.omic_names = []
        self.n_samples = 0
        self.availability_mask = None
        
        # XAI inferrers (one per omic that can be missing)
        self.inferrers: Dict[str, MissingOmicInferrer] = {}
        
        # Results
        self.results = {}
        
        logger.info(f"Initialized UseGroupsExperiment")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Toy mode: {self.toy_mode}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def load_data(self):
        """Load coral multi-omics data."""
        data_config = self.config.get("data", {})
        data_dir = Path(data_config.get("processed_dir", "data/processed"))
        
        logger.info(f"Loading data from {data_dir}")
        
        # Load each omic type, keeping track of sample IDs
        self.omic_dfs = {}
        
        for omic_file in data_dir.glob("*.tsv"):
            omic_name = omic_file.stem
            
            # Skip inferred and metadata files
            if omic_name.startswith("inferred_") or omic_name.endswith("_metadata"):
                continue
            if omic_name.endswith("_filtered"):  # Skip filtered versions
                continue
            
            df = pd.read_csv(omic_file, sep='\t', index_col=0)
            self.omic_dfs[omic_name] = df
            logger.info(f"  Loaded {omic_name}: {df.shape}")
        
        self.omic_names = list(self.omic_dfs.keys())
        
        # Find common samples across ALL omics
        sample_sets = [set(df.index) for df in self.omic_dfs.values()]
        common_samples = set.intersection(*sample_sets) if sample_sets else set()
        
        logger.info(f"  Common samples across all omics: {len(common_samples)}")
        
        # If no common samples, use union with availability mask
        if len(common_samples) == 0:
            all_samples = sorted(set.union(*sample_sets))
            logger.warning(f"  No common samples! Using union: {len(all_samples)} total samples")
            
            # Create aligned data with NaN for missing
            self.sample_ids = all_samples
            self.data = {}
            for omic_name, df in self.omic_dfs.items():
                aligned = df.reindex(all_samples)
                self.data[omic_name] = aligned.values.astype(np.float32)
        else:
            # Use common samples
            common_samples = sorted(common_samples)
            self.sample_ids = common_samples
            self.data = {}
            for omic_name, df in self.omic_dfs.items():
                self.data[omic_name] = df.loc[common_samples].values.astype(np.float32)
        
        self.n_samples = len(self.sample_ids)
        
        # Load or generate labels based on sample IDs
        labels_file = data_dir / "labels.tsv"
        if labels_file.exists():
            labels_df = pd.read_csv(labels_file, sep='\t', index_col=0)
            self.labels = labels_df.reindex(self.sample_ids).values.flatten().astype(np.int64)
        else:
            # Try to infer labels from sample names (e.g., "Amb" vs "HiT" for ambient vs heat)
            self.labels = self._infer_labels_from_sample_names()
        
        # Create availability mask
        self._create_availability_mask()
        
        # Toy mode: use subset
        if self.toy_mode:
            n_toy = min(20, self.n_samples)
            logger.info(f"TOY MODE: Using first {n_toy} samples")
            
            for omic in self.data:
                self.data[omic] = self.data[omic][:n_toy]
            
            self.labels = self.labels[:n_toy]
            self.sample_ids = self.sample_ids[:n_toy]
            self.n_samples = n_toy
            self._create_availability_mask()
        
        logger.info(f"Total samples: {self.n_samples}")
        logger.info(f"Omics: {self.omic_names}")
    
    def _infer_labels_from_sample_names(self) -> np.ndarray:
        """
        Infer treatment labels from sample names.
        
        Sample naming convention in Williams et al. data:
        - MC-289_T1-Amb_1603 -> Ambient (control)
        - MC-289_T1-HiT_2878 -> High Temperature (stress)
        - MC-289_Field_289-1 -> Field sample
        
        Returns:
            Array of labels (0 = ambient/field, 1 = heat stress)
        """
        labels = []
        for sample_id in self.sample_ids:
            sample_str = str(sample_id)
            if 'HiT' in sample_str or 'Heat' in sample_str:
                labels.append(1)  # Heat stress
            elif 'Amb' in sample_str or 'Field' in sample_str:
                labels.append(0)  # Ambient/control
            else:
                # Default to 0 for unknown
                labels.append(0)
        
        labels_arr = np.array(labels, dtype=np.int64)
        logger.info(f"Inferred labels from sample names: {np.bincount(labels_arr)} (0=ambient, 1=heat)")
        return labels_arr
    
    def _create_availability_mask(self):
        """Create mask indicating which omics are available per sample."""
        # For now, assume all samples have all omics
        # In real data, some samples may be missing certain omics
        self.availability_mask = np.ones((self.n_samples, len(self.omic_names)), dtype=np.float32)
        
        # Check for NaN values indicating missing data
        for i, omic in enumerate(self.omic_names):
            if omic in self.data:
                nan_samples = np.isnan(self.data[omic]).any(axis=1)
                self.availability_mask[nan_samples, i] = 0
    
    def train_xai_inferrers(self, train_idx: Optional[np.ndarray] = None):
        """
        Train XAI inferrers for each omic type.
        
        Args:
            train_idx: Indices of training samples to use. If None, uses all
                samples (DEPRECATED — causes data leakage in CV settings).
                Always pass fold-specific training indices to prevent leakage.
        """
        if train_idx is None:
            logger.warning(
                "train_xai_inferrers called without train_idx — using all "
                "samples. This causes data leakage if used before CV splits."
            )
        
        logger.info("Training XAI inferrers for missing omic inference...")
        
        xai_config = self.config.get("xai", {})
        latent_dim = xai_config.get("latent_dim", 64)
        surrogate_type = xai_config.get("surrogate_type", "lightgbm")
        
        # Subset data to training indices only (prevents data leakage)
        if train_idx is not None:
            fold_data = {omic: self.data[omic][train_idx] for omic in self.omic_names}
            fold_labels = self.labels[train_idx]
        else:
            fold_data = self.data
            fold_labels = self.labels
        
        # Train an inferrer for each omic
        for target_omic in self.omic_names:
            logger.info(f"Training inferrer for {target_omic}...")
            
            try:
                inferrer, metrics = train_inference_model(
                    data_dict=fold_data,
                    target_omic=target_omic,
                    labels=fold_labels,
                    latent_dim=latent_dim,
                    surrogate_type=surrogate_type,
                    seed=self.seed,
                    output_dir=self.output_dir / "models" / "xai",
                )
                
                self.inferrers[target_omic] = inferrer
                logger.info(f"  {target_omic} inferrer trained: {metrics}")
            
            except Exception as e:
                logger.warning(f"  Failed to train inferrer for {target_omic}: {e}")
    
    def _get_data_for_strategy(
        self,
        strategy: str,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        """
        Prepare data according to missing data handling strategy.
        
        Returns:
            Tuple of (train_data, test_data, train_labels, test_labels)
        """
        train_data = {}
        test_data = {}
        
        if strategy == "complete_case":
            # Only use samples with all omics
            complete_mask = self.availability_mask.all(axis=1)
            train_complete = np.intersect1d(train_idx, np.where(complete_mask)[0])
            test_complete = np.intersect1d(test_idx, np.where(complete_mask)[0])
            
            for omic in self.omic_names:
                train_data[omic] = self.data[omic][train_complete]
                test_data[omic] = self.data[omic][test_complete]
            
            train_labels = self.labels[train_complete]
            test_labels = self.labels[test_complete]
        
        elif strategy == "zero_imputation":
            # Replace missing with zeros
            for omic in self.omic_names:
                train_d = self.data[omic][train_idx].copy()
                test_d = self.data[omic][test_idx].copy()
                train_d[np.isnan(train_d)] = 0
                test_d[np.isnan(test_d)] = 0
                train_data[omic] = train_d
                test_data[omic] = test_d
            
            train_labels = self.labels[train_idx]
            test_labels = self.labels[test_idx]
        
        elif strategy == "mean_imputation":
            # Replace missing with mean
            for omic in self.omic_names:
                train_d = self.data[omic][train_idx].copy()
                test_d = self.data[omic][test_idx].copy()
                
                # Compute mean from non-nan values
                mean_vals = np.nanmean(train_d, axis=0)
                
                # Fill NaN with mean
                for j in range(train_d.shape[1]):
                    train_d[np.isnan(train_d[:, j]), j] = mean_vals[j]
                    test_d[np.isnan(test_d[:, j]), j] = mean_vals[j]
                
                train_data[omic] = train_d
                test_data[omic] = test_d
            
            train_labels = self.labels[train_idx]
            test_labels = self.labels[test_idx]
        
        elif strategy == "mosaic_mask":
            # Use availability masking (model handles missing)
            for omic in self.omic_names:
                train_d = self.data[omic][train_idx].copy()
                test_d = self.data[omic][test_idx].copy()
                train_d[np.isnan(train_d)] = 0
                test_d[np.isnan(test_d)] = 0
                train_data[omic] = train_d
                test_data[omic] = test_d
            
            # Include masks
            train_data["_mask"] = self.availability_mask[train_idx]
            test_data["_mask"] = self.availability_mask[test_idx]
            
            train_labels = self.labels[train_idx]
            test_labels = self.labels[test_idx]
        
        elif strategy == "xai_inferred":
            # Use XAI-inferred representations for missing omics
            for omic in self.omic_names:
                train_d = self.data[omic][train_idx].copy()
                test_d = self.data[omic][test_idx].copy()
                
                # For samples missing this omic, use inferred representation
                if omic in self.inferrers:
                    # Find samples missing this omic
                    train_missing = np.isnan(train_d).any(axis=1)
                    test_missing = np.isnan(test_d).any(axis=1)
                    
                    if train_missing.any():
                        # Infer from available omics
                        available = {
                            o: self.data[o][train_idx[train_missing]] 
                            for o in self.omic_names if o != omic
                        }
                        # Replace NaN in available omics with 0 for inference
                        for k in available:
                            available[k] = np.nan_to_num(available[k], nan=0.0)
                        inferred = self.inferrers[omic].infer(available)
                        # Pad inferred latent to match original dimensions
                        n_missing = inferred.shape[0]
                        original_dim = train_d.shape[1]
                        if inferred.shape[1] < original_dim:
                            padded = np.zeros((n_missing, original_dim))
                            padded[:, :inferred.shape[1]] = inferred
                            inferred = padded
                        train_d[train_missing] = inferred[:, :original_dim]
                    
                    if test_missing.any():
                        available = {
                            o: self.data[o][test_idx[test_missing]]
                            for o in self.omic_names if o != omic
                        }
                        # Replace NaN in available omics with 0 for inference
                        for k in available:
                            available[k] = np.nan_to_num(available[k], nan=0.0)
                        inferred = self.inferrers[omic].infer(available)
                        # Pad inferred latent to match original dimensions
                        n_missing = inferred.shape[0]
                        original_dim = test_d.shape[1]
                        if inferred.shape[1] < original_dim:
                            padded = np.zeros((n_missing, original_dim))
                            padded[:, :inferred.shape[1]] = inferred
                            inferred = padded
                        test_d[test_missing] = inferred[:, :original_dim]
                
                # Fill any remaining NaN with zeros
                train_d[np.isnan(train_d)] = 0
                test_d[np.isnan(test_d)] = 0
                
                train_data[omic] = train_d
                test_data[omic] = test_d
            
            train_labels = self.labels[train_idx]
            test_labels = self.labels[test_idx]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return train_data, test_data, train_labels, test_labels
    
    def create_model(self, **kwargs) -> nn.Module:
        """
        Create model (required by BaseExperiment).
        
        Args:
            **kwargs: Must include 'strategy' key
            
        Returns:
            Model instance
        """
        strategy = kwargs.get("strategy", "xai_inferred")
        return self._create_model(strategy)
    
    def _create_model(self, strategy: str) -> nn.Module:
        """Create model for given strategy."""
        model_config = self.config.get("model", {})
        
        # Get input dimensions
        omics_dims = {omic: self.data[omic].shape[1] for omic in self.omic_names}
        
        if strategy in ["mosaic_mask", "xai_inferred"]:
            # Use mosaic VFL model
            model = MosaicVFLModel(
                omics_dims=omics_dims,
                encoder_hidden_dim=model_config.get("encoder_hidden_dim", 128),
                embedding_dim=model_config.get("embedding_dim", 64),
                fusion_hidden_dim=model_config.get("fusion_hidden_dim", 128),
                n_classes=2,
                dropout=model_config.get("dropout", 0.3),
            )
        else:
            # Standard late fusion
            model = LateFusionModel(
                omics_dims=omics_dims,
                encoder_hidden_dim=model_config.get("encoder_hidden_dim", 128),
                embedding_dim=model_config.get("embedding_dim", 64),
                fusion_hidden_dim=model_config.get("fusion_hidden_dim", 128),
                n_classes=2,
                dropout=model_config.get("dropout", 0.3),
            )
        
        return model.to(self.device)
    
    def _train_model(
        self,
        model: nn.Module,
        train_data: Dict[str, np.ndarray],
        train_labels: np.ndarray,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> nn.Module:
        """Train model."""
        training_config = self.config.get("training", {})
        n_epochs = training_config.get("n_epochs", 50)
        batch_size = training_config.get("batch_size", 32)
        lr = training_config.get("learning_rate", 1e-3)
        
        if self.toy_mode:
            n_epochs = min(10, n_epochs)
        
        # Create data loaders
        train_tensors = [
            torch.tensor(train_data[omic], dtype=torch.float32)
            for omic in self.omic_names
        ]
        train_y = torch.tensor(train_labels, dtype=torch.long)
        
        train_dataset = TensorDataset(*train_tensors, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            
            for batch in train_loader:
                *X_batch, y_batch = batch
                
                # Create input dict
                X_dict = {omic: X_batch[i].to(self.device) 
                         for i, omic in enumerate(self.omic_names)}
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_dict)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        return model
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_data: Dict[str, np.ndarray],
        test_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate model and return metrics."""
        model.eval()
        
        # Create test tensors
        test_tensors = [
            torch.tensor(test_data[omic], dtype=torch.float32).to(self.device)
            for omic in self.omic_names
        ]
        test_y = torch.tensor(test_labels, dtype=torch.long)
        
        X_dict = {omic: test_tensors[i] for i, omic in enumerate(self.omic_names)}
        
        with torch.no_grad():
            outputs = model(X_dict)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
        
        y_true = test_y.numpy()
        
        metrics = compute_classification_metrics(y_true, preds, probs)
        
        return {
            "y_true": y_true,
            "y_pred": preds,
            "y_prob": probs,
            "metrics": metrics,
        }
    
    def run_strategy(
        self,
        strategy: str,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> MetricsTracker:
        """Run experiment for a specific strategy."""
        logger.info(f"Running strategy: {strategy}")
        
        tracker = MetricsTracker(
            experiment_name=f"h2_{strategy}",
            task="classification",
            output_dir=self.output_dir / "metrics",
        )
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            logger.info(f"  Fold {fold_idx + 1}/{len(cv_splits)}")
            
            # For XAI-inferred strategy, retrain inferrers on this fold's
            # training data only to prevent data leakage from test samples.
            if strategy == "xai_inferred":
                self.train_xai_inferrers(train_idx=train_idx)
            
            # Get data for this strategy
            train_data, test_data, train_labels, test_labels = self._get_data_for_strategy(
                strategy, train_idx, test_idx
            )
            
            # Skip if no data (complete_case with no complete samples)
            if len(train_labels) == 0 or len(test_labels) == 0:
                logger.warning(f"  Skipping fold {fold_idx}: no samples")
                continue
            
            logger.info(f"    Train: {len(train_labels)}, Test: {len(test_labels)}")
            
            # Create and train model
            model = self._create_model(strategy)
            model = self._train_model(model, train_data, train_labels)
            
            # Evaluate
            results = self._evaluate_model(model, test_data, test_labels)
            
            tracker.add_fold(
                fold_idx,
                results["y_true"],
                results["y_pred"],
                results["y_prob"],
                additional_info={
                    "n_train": len(train_labels),
                    "n_test": len(test_labels),
                }
            )
        
        tracker.save()
        return tracker
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full H2 experiment.
        
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("Experiment H2: Use Groups with XAI Inference")
        logger.info("=" * 60)
        
        # Load data
        self.load_data()
        
        # NOTE: XAI inferrers are trained per-fold inside run_strategy()
        # to prevent data leakage. Do NOT call train_xai_inferrers() here.
        
        # Create CV splits using sklearn directly
        n_folds = self.config.get("cv", {}).get("n_folds", 5)
        if self.toy_mode:
            n_folds = min(2, self.n_samples)  # Can't have more folds than samples
        
        # Ensure we have enough samples for CV
        n_folds = min(n_folds, self.n_samples)
        
        from sklearn.model_selection import StratifiedKFold, KFold
        
        if len(np.unique(self.labels)) > 1:
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            cv_splits = list(kfold.split(np.arange(self.n_samples), self.labels))
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            cv_splits = list(kfold.split(np.arange(self.n_samples)))
        
        logger.info(f"Using {n_folds}-fold CV on {self.n_samples} samples")
        
        # Run all strategies
        strategies = self.config.get("strategies", self.STRATEGIES)
        trackers = {}
        
        for strategy in strategies:
            try:
                trackers[strategy] = self.run_strategy(strategy, cv_splits)
            except Exception as e:
                logger.error(f"Strategy {strategy} failed: {e}")
                continue
        
        # Aggregate results
        self.results = self._aggregate_results(trackers)
        
        # Statistical comparison
        self.results["comparison"] = self._statistical_comparison(trackers)
        
        # Generate outputs
        self._save_results()
        self._generate_plots(trackers)
        self._generate_report()
        
        return self.results
    
    def _aggregate_results(self, trackers: Dict[str, MetricsTracker]) -> Dict:
        """Aggregate results from all strategies."""
        results = {
            "experiment": "h2_use_groups",
            "timestamp": datetime.now().isoformat(),
            "n_samples": self.n_samples,
            "omics": self.omic_names,
            "strategies": {},
        }
        
        for name, tracker in trackers.items():
            agg = tracker.get_aggregated()
            results["strategies"][name] = {
                "metrics": {
                    k: {"mean": float(v["mean"]), "std": float(v["std"])}
                    for k, v in agg.items()
                },
                "n_folds": len(tracker.folds),
            }
        
        return results
    
    def _statistical_comparison(self, trackers: Dict[str, MetricsTracker]) -> Dict:
        """Perform statistical comparison between strategies."""
        auroc_scores = {}
        
        for name, tracker in trackers.items():
            scores = [f["metrics"]["auroc"] for f in tracker.folds]
            if scores:  # Only include strategies with results
                auroc_scores[name] = scores
        
        if len(auroc_scores) < 2:
            logger.warning("Not enough strategies with results for comparison")
            return {}
        
        # Find a valid baseline (one with results)
        baseline = None
        for candidate in ["complete_case", "zero_imputation", "mean_imputation"]:
            if candidate in auroc_scores and auroc_scores[candidate]:
                baseline = candidate
                break
        
        if baseline is None:
            baseline = list(auroc_scores.keys())[0]
        
        # Ensure all score arrays have the same length as baseline
        baseline_len = len(auroc_scores[baseline])
        filtered_scores = {baseline: auroc_scores[baseline]}
        for name, scores in auroc_scores.items():
            if name != baseline and len(scores) == baseline_len:
                filtered_scores[name] = scores
        
        if len(filtered_scores) < 2:
            logger.warning("Not enough strategies with matching fold counts for comparison")
            return {"note": "Strategies have different fold counts"}
        
        try:
            comparison = compare_models(
                filtered_scores,
                metric="auroc",
                baseline=baseline,
                test="wilcoxon",
                correction="bonferroni",
            )
            logger.info("\n" + format_comparison_table(comparison))
            return comparison
        except Exception as e:
            logger.warning(f"Statistical comparison failed: {e}")
            return {"error": str(e)}
    
    def _save_results(self):
        """Save results to disk."""
        # Save main results
        results_file = self.output_dir / "metrics.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_file}")
        
        # Save comparison table as CSV
        if "comparison" in self.results:
            comparison_data = self.results["comparison"]
            try:
                # Try to extract a flat table from the comparison results
                if isinstance(comparison_data, dict) and "models" in comparison_data:
                    # Standard compare_models output format
                    rows = []
                    for model_name, model_info in comparison_data["models"].items():
                        row = {"strategy": model_name}
                        if isinstance(model_info, dict):
                            row.update({k: v for k, v in model_info.items() if not isinstance(v, (dict, list))})
                        rows.append(row)
                    if rows:
                        comparison_df = pd.DataFrame(rows)
                        comparison_df.to_csv(self.output_dir / "comparison_table.csv", index=False)
                elif isinstance(comparison_data, dict) and "error" not in comparison_data:
                    # Try to save as JSON instead
                    with open(self.output_dir / "comparison.json", "w") as f:
                        json.dump(comparison_data, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Could not save comparison table: {e}")
    
    def _generate_plots(self, trackers: Dict[str, MetricsTracker]):
        """Generate visualization plots."""
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # Filter strategies that have valid metrics
            strategies = []
            auroc_means = []
            auroc_stds = []
            
            for s in self.results.get("strategies", {}).keys():
                strategy_data = self.results["strategies"][s]
                if (isinstance(strategy_data, dict) and 
                    "metrics" in strategy_data and 
                    isinstance(strategy_data["metrics"], dict) and
                    "auroc" in strategy_data["metrics"] and
                    isinstance(strategy_data["metrics"]["auroc"], dict) and
                    "mean" in strategy_data["metrics"]["auroc"]):
                    strategies.append(s)
                    auroc_means.append(strategy_data["metrics"]["auroc"]["mean"])
                    auroc_stds.append(strategy_data["metrics"]["auroc"].get("std", 0))
            
            if not strategies:
                logger.warning("No strategies with valid metrics for plotting")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(strategies))
            bars = ax.bar(x, auroc_means, yerr=auroc_stds, capsize=5, alpha=0.8)
            
            # Highlight XAI strategy
            for i, s in enumerate(strategies):
                if "xai" in s.lower():
                    bars[i].set_color("green")
            
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha="right")
            ax.set_ylabel("AUROC")
            ax.set_title("H2: Missing Data Handling Strategies Comparison")
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(figures_dir / "auroc_comparison.png", dpi=300)
            plt.close()
            
            logger.info(f"Saved figures to {figures_dir}")
        
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    def _generate_report(self):
        """Generate results report template."""
        reports_dir = self.output_dir.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report = f"""# H2 Experiment Results: Use Groups with XAI Inference

## Overview
- **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Dataset:** Williams et al. (2021) Montipora capitata thermal stress
- **DOI:** 10.7717/peerj.12335
- **Samples:** {self.n_samples}
- **Omics:** {', '.join(self.omic_names)}

## Strategies Compared
1. **Complete-case:** Only samples with all omics
2. **Zero imputation:** Missing → zeros
3. **Mean imputation:** Missing → global mean
4. **Mosaic mask:** Use availability masks
5. **XAI-inferred:** Gradient saliency-based proxy for missing omics

## Results Summary

| Strategy | AUROC (mean ± std) |
|----------|-------------------|
"""
        
        for strategy, data in self.results.get("strategies", {}).items():
            metrics = data.get("metrics", {})
            if "auroc" in metrics:
                auroc = metrics["auroc"]
                report += f"| {strategy} | {auroc['mean']:.4f} ± {auroc['std']:.4f} |\n"
        
        report += """
## Key Findings

_[To be filled after analysis]_

## Conclusions

_[To be filled after analysis]_

## Files Generated
- `outputs/h2/metrics.json` - Full metrics
- `outputs/h2/figures/auroc_comparison.png` - Performance plot
- `outputs/h2/comparison_table.csv` - Statistical comparison
- `outputs/models/xai/` - XAI inference models
"""
        
        report_file = reports_dir / "h2_results_template.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"Saved report template to {report_file}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run H2 Use Groups experiment with XAI inference"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/h2_config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/h2"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Run in toy mode with minimal data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load config
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    config["seed"] = args.seed
    
    # Run experiment
    experiment = UseGroupsExperiment(
        config=config,
        output_dir=args.output,
        device=args.device,
        toy_mode=args.toy,
    )
    
    results = experiment.run()
    
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
