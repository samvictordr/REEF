#!/usr/bin/env python3
"""
Mosaic-VFL Adapter for Coral Multi-Omics Benchmark.

Wraps the existing Mosaic-VFL implementation to provide a unified interface
for fair comparison with LASER-VFL.

This adapter:
1. Uses the same canonical data format as the LASER adapter
2. Provides train/evaluate/predict methods with identical signatures
3. Ensures comparable hyperparameters and training procedures

Key parity constraints:
- Same preprocessed data (no value modifications)
- Same train/val/test splits (loaded from JSON)
- Matched model capacity (hidden_dim, embedding_dim)
- Same optimization budget (epochs, patience)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vertical_federated import MosaicVFLModel, VFLModel

logger = logging.getLogger(__name__)


class CoralOmicsDataset(Dataset):
    """
    Dataset for coral multi-omics compatible with Mosaic-VFL.
    
    Provides data as dict of omics tensors + labels + availability mask.
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        labels: np.ndarray,
        omic_names: List[str],
        sample_indices: np.ndarray,
        p_miss: float = 0.0,
    ):
        """
        Initialize dataset.
        
        Args:
            data: Dict mapping omics name -> (n_samples, n_features) array
            labels: Array of labels (n_samples,)
            omic_names: Ordered list of omics names
            sample_indices: Indices to use from full dataset
            p_miss: Probability of missing omics (for mosaic training)
        """
        self.omic_names = omic_names
        self.n_omics = len(omic_names)
        self.p_miss = p_miss
        
        # Extract samples for this split
        self.labels = labels[sample_indices].astype(np.int64)
        self.features = {}
        for name in omic_names:
            arr = data[name][sample_indices].astype(np.float32)
            self.features[name] = torch.tensor(arr, dtype=torch.float32)
        
        self.n_samples = len(sample_indices)
        
        # Generate availability masks
        self._generate_masks()
    
    def _generate_masks(self):
        """Generate availability masks for mosaic-style missingness."""
        if self.p_miss == 0.0:
            # All omics always available
            self.masks = [torch.ones(self.n_omics, dtype=torch.float32)
                         for _ in range(self.n_samples)]
        else:
            # Bernoulli missingness per omics
            self.masks = []
            for _ in range(self.n_samples):
                mask = (np.random.rand(self.n_omics) > self.p_miss).astype(np.float32)
                # Ensure at least one omics is present
                if mask.sum() == 0:
                    mask[np.random.randint(self.n_omics)] = 1.0
                self.masks.append(torch.tensor(mask, dtype=torch.float32))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Return dict of omics tensors, label, and mask.
        """
        features = {name: self.features[name][idx] for name in self.omic_names}
        label = self.labels[idx]
        mask = self.masks[idx]
        
        return features, label, mask


def collate_fn(batch):
    """
    Collate function for Mosaic-VFL.
    
    Returns dict of stacked omics tensors, labels, masks.
    """
    features_list, labels_list, masks_list = zip(*batch)
    
    # Stack features per omics
    omic_names = list(features_list[0].keys())
    stacked_features = {
        name: torch.stack([f[name] for f in features_list])
        for name in omic_names
    }
    
    labels = torch.tensor(labels_list, dtype=torch.long)
    masks = torch.stack(masks_list)
    
    return stacked_features, labels, masks


class MosaicAdapter:
    """
    Adapter to run Mosaic-VFL on coral multi-omics data.
    
    Provides unified interface matching LASERAdapter.
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        labels: np.ndarray,
        omic_names: List[str],
        config: Dict[str, Any],
        device: torch.device = None,
    ):
        """
        Initialize adapter.
        
        Args:
            data: Dict of omics data arrays
            labels: Label array
            omic_names: Ordered list of omics names
            config: Configuration dict with hyperparameters
            device: Torch device
        """
        # Standardize data for numerical stability (same as LASER adapter)
        self.data = {}
        self.scalers = {}
        for name, arr in data.items():
            # Standard scaling: (x - mean) / std
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True) + 1e-8  # Avoid div by zero
            self.data[name] = ((arr - mean) / std).astype(np.float32)
            self.scalers[name] = {"mean": mean, "std": std}
        
        self.labels = labels
        self.omic_names = omic_names
        self.n_omics = len(omic_names)
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Model and training state
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def _get_omics_dims(self) -> Dict[str, int]:
        """Get feature dimension for each omics."""
        return {name: self.data[name].shape[1] for name in self.omic_names}
    
    def _create_model(self):
        """
        Create Mosaic-VFL model for coral data.
        """
        omics_dims = self._get_omics_dims()
        
        hidden_dim = self.config.get("hidden_dim", 128)
        embedding_dim = self.config.get("embedding_dim", 64)
        fusion_hidden = self.config.get("fusion_hidden_dim", 128)
        # For binary classification with BCEWithLogitsLoss, use n_classes=1
        n_classes = self.config.get("n_classes", 1)
        dropout = self.config.get("dropout", 0.3)
        fusion_method = self.config.get("fusion_method", "concat")
        use_batchnorm = self.config.get("use_batchnorm", True)
        
        # Use MosaicVFLModel which handles missing omics
        self.model = MosaicVFLModel(
            omics_dims=omics_dims,
            encoder_hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            fusion_hidden_dim=fusion_hidden,
            n_classes=n_classes,
            task="classification",
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            fusion_method=fusion_method,
        ).to(self.device)
        
        return self.model
    
    def _create_optimizer(self):
        """Create optimizer."""
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        return self.optimizer
    
    def _create_criterion(self):
        """Create loss criterion."""
        self.criterion = nn.BCEWithLogitsLoss()
        return self.criterion
    
    def create_dataloader(
        self,
        sample_indices: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        p_miss: float = 0.0,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create DataLoader for given sample indices.
        """
        dataset = CoralOmicsDataset(
            data=self.data,
            labels=self.labels,
            omic_names=self.omic_names,
            sample_indices=sample_indices,
            p_miss=p_miss,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=drop_last,
        )
        
        return loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for features, targets, mask in train_loader:
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)
            mask = mask.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward with mask
            logits = self.model(features, mask=mask)
            
            # Handle output shape
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            
            loss = self.criterion(logits, targets.float())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(targets)
            
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == targets).sum().item()
            total_samples += len(targets)
        
        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }
    
    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        """
        self.model.eval()
        
        all_probs = []
        all_targets = []
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, targets, mask in test_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                
                logits = self.model(features, mask=mask)
                
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                
                loss = self.criterion(logits, targets.float())
                
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                
                total_loss += loss.item() * len(targets)
                total_correct += (preds == targets).sum().item()
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                total_samples += len(targets)
        
        all_probs = np.concatenate(all_probs) if all_probs else np.array([])
        all_targets = np.concatenate(all_targets) if all_targets else np.array([])
        
        results = {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }
        
        if len(all_probs) > 0:
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                f1_score, brier_score_loss
            )
            
            preds_binary = (all_probs >= 0.5).astype(int)
            
            try:
                results["auroc"] = float(roc_auc_score(all_targets, all_probs))
            except ValueError:
                results["auroc"] = 0.5
            
            try:
                results["auprc"] = float(average_precision_score(all_targets, all_probs))
            except ValueError:
                results["auprc"] = float(np.mean(all_targets))
            
            results["f1"] = float(f1_score(all_targets, preds_binary, zero_division=0))
            results["brier"] = float(brier_score_loss(all_targets, all_probs))
        
        if return_predictions:
            results["predictions"] = all_probs
            results["targets"] = all_targets
        
        return results
    
    def train(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        p_miss_train: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.
        """
        self._create_model()
        self._create_optimizer()
        self._create_criterion()
        
        # Use drop_last=True for training to avoid batch size=1 issues with BatchNorm
        train_loader = self.create_dataloader(
            train_indices, batch_size, shuffle=True, p_miss=p_miss_train, drop_last=True
        )
        val_loader = self.create_dataloader(
            val_indices, batch_size, shuffle=False, p_miss=0.0, drop_last=False
        )
        
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        history = {"train": [], "val": []}
        
        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)
            
            val_auroc = val_metrics.get("auroc", 0.5)
            
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        history["best_val_auroc"] = best_val_auroc
        history["best_epoch"] = len(history["val"]) - patience_counter
        
        return history
    
    def predict(
        self,
        test_indices: np.ndarray,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for test samples.
        """
        test_loader = self.create_dataloader(
            test_indices, batch_size, shuffle=False, p_miss=0.0
        )
        results = self.evaluate(test_loader, return_predictions=True)
        return results["predictions"], results["targets"]
    
    def save_checkpoint(self, path: Path, metadata: Dict[str, Any] = None):
        """
        Save model checkpoint with full reproducibility info.
        
        Args:
            path: Path to save checkpoint (.pt)
            metadata: Additional metadata to save
        """
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "omic_names": self.omic_names,
            "omic_dims": self._get_omics_dims(),
            "config": self.config,
            "scalers": self.scalers,
        }
        
        if metadata:
            checkpoint["metadata"] = metadata
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint (.pt)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model if needed
        if self.model is None:
            self._create_model()
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Restore scalers if available
        if "scalers" in checkpoint:
            self.scalers = checkpoint["scalers"]
        
        logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint.get("metadata", {})
    
    def get_concatenated_features(
        self, 
        sample_indices: np.ndarray = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get concatenated features for XAI pipeline.
        
        Args:
            sample_indices: Indices to include (all if None)
        
        Returns:
            Tuple of (X array, feature_names list)
        """
        if sample_indices is None:
            sample_indices = np.arange(len(self.labels))
        
        # Concatenate omics in order
        X_parts = []
        feature_names = []
        
        for omic in self.omic_names:
            arr = self.data[omic][sample_indices]
            X_parts.append(arr)
            
            n_features = arr.shape[1]
            feature_names.extend([f"{omic}_{i}" for i in range(n_features)])
        
        X = np.concatenate(X_parts, axis=1)
        
        return X, feature_names


def load_coral_data(data_dir: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load coral multi-omics data in canonical format.
    (Same implementation as laser_adapter for consistency)
    
    Data format: rows=samples, first column='sample_id', rest are features.
    """
    omics_files = {
        "proteomics": data_dir / "proteomics.tsv",
        "metabolomics": data_dir / "metabolomics.tsv",
        "microbiome": data_dir / "microbiome.tsv",
        "transcriptomics": data_dir / "transcriptomics.tsv",
    }
    
    data = {}
    sample_ids = None
    
    for name, path in omics_files.items():
        if path.exists():
            df = pd.read_csv(path, sep='\t')
            
            # First column is sample_id, rest are features
            if 'sample_id' in df.columns:
                ids = df['sample_id'].tolist()
                df = df.drop(columns=['sample_id'])
            else:
                ids = df.iloc[:, 0].tolist()
                df = df.iloc[:, 1:]
            
            if sample_ids is None:
                sample_ids = ids
            else:
                # Align to common samples
                common = list(set(sample_ids) & set(ids))
                sample_ids = [s for s in sample_ids if s in common]
            
            # Convert to numpy, fill NaN with 0
            arr = df.values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            data[name] = arr
    
    # Re-align all arrays to common sample order
    if sample_ids:
        for name, path in omics_files.items():
            if path.exists() and name in data:
                df = pd.read_csv(path, sep='\t')
                if 'sample_id' in df.columns:
                    df = df.set_index('sample_id')
                else:
                    df = df.set_index(df.columns[0])
                # Select common samples in order
                df = df.loc[sample_ids]
                arr = df.values.astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                data[name] = arr
    
    return data, sample_ids


if __name__ == "__main__":
    # Simple test
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    print("Loading data...")
    data, sample_ids = load_coral_data(data_dir)
    print(f"Loaded {len(data)} omics for {len(sample_ids)} samples")
    
    labels = np.array([1 if "HiT" in s else 0 for s in sample_ids])
    print(f"Labels: {np.bincount(labels)}")
    
    config = {
        "hidden_dim": 64,
        "embedding_dim": 32,
        "n_classes": 2,
        "lr": 1e-3,
    }
    
    adapter = MosaicAdapter(
        data=data,
        labels=labels,
        omic_names=list(data.keys()),
        config=config,
    )
    
    train_idx = np.arange(min(10, len(sample_ids) - 3))
    val_idx = np.arange(min(10, len(sample_ids) - 3), len(sample_ids))
    
    print("\nTraining Mosaic adapter...")
    history = adapter.train(
        train_idx, val_idx,
        n_epochs=5,
        batch_size=4,
    )
    print(f"Best val AUROC: {history['best_val_auroc']:.4f}")
