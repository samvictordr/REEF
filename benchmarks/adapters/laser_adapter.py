#!/usr/bin/env python3
"""
LASER-VFL Adapter for Coral Multi-Omics Benchmark (Stabilized Version).

This adapter wraps LASER-VFL for fair comparison with Mosaic-VFL on small-N,
high-P coral multi-omics data. It includes stabilization measures to prevent
exploding gradients, saturated softmax, and numerical instability.

IMPORTANT: This adapter uses the ORIGINAL LASER model architecture from
LASER-VFL-master/models/mlp.py (MLPFeatureExtractor, BaseLaserModel) with
MINIMAL stabilization wrappers. This ensures a fair comparison while addressing
the numerical instability that occurs in the extreme P>>N regime.

STABILIZATION MEASURES (see benchmarks/laser_patch_changelog.md):
1. AdamW optimizer with lower learning rate (1e-4) and weight decay (1e-4)
2. Gradient clipping (max_norm=1.0) after backward() before step()
3. LayerNorm wrapper around original MLP when batch_size < 8
4. Small weight initialization for classification heads (std=1e-3, bias=0)
5. Max logit monitoring with early halt if |logit| > 1e4
6. Early stopping on validation loss (patience=10)
7. Per-epoch logging: train_loss, val_loss, max_grad_norm, max_logit, lr

LASER-VFL expects:
- Features split into num_clients blocks (one per omics)
- CustomDataset wrapper that provides mask patterns
- DataLoader with custom collate_fn

This adapter:
1. Loads preprocessed omics data as canonical arrays
2. Applies StandardScaler normalization (fit on train only)
3. Converts to LASER's expected input format
4. Provides train/test functions with stability monitoring
5. Returns predictions in standardized format for fair comparison

Key design decisions for parity:
- Uses exact same preprocessed matrices as Mosaic-VFL
- Only reshapes/renames, never modifies underlying values
- Enforces identical train/val/test splits via split file
- Same scaling strategy as Mosaic adapter (StandardScaler on train)
"""

import sys
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

# Add LASER-VFL to path to import original model classes
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LASER_ROOT = PROJECT_ROOT / "LASER-VFL-master"
sys.path.insert(0, str(LASER_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# IMPORT ORIGINAL LASER MODEL CLASSES
# ============================================================================
# These are the authentic LASER model components from the original LASER-VFL
# repository. We import them directly to ensure architectural fidelity.
from models.mlp import MLPFeatureExtractor  # Original LASER feature extractor
from models.model_utils import FusionModel, BaseLaserModel  # Original LASER core

logger = logging.getLogger(__name__)


# ==============================================================================
# STABILIZATION CONFIGURATION
# ==============================================================================

@dataclass
class StabilizationConfig:
    """
    Configuration for LASER training stabilization.
    
    These parameters address numerical instability on small-N, high-P data.
    Defaults are tuned for coral multi-omics (N~14, P~100K combined features).
    """
    # Optimizer
    lr: float = 1e-4                           # Lower LR prevents overshooting
    weight_decay: float = 1e-4                 # L2 regularization
    use_adamw: bool = True                     # AdamW vs Adam
    
    # Gradient clipping (prevents exploding gradients)
    clip_grad_norm: float = 1.0                # Max gradient L2 norm
    
    # Weight initialization (small init prevents large initial logits)
    init_std: float = 1e-3                     # Std for head weights
    init_bias_zero: bool = True                # Zero-init biases
    
    # Normalization layer handling
    disable_batchnorm_if_batchsize_lt: int = 8 # Threshold for switching to LayerNorm
    use_layernorm: bool = True                 # Use LayerNorm in small-batch regime
    
    # Logit monitoring (detects numerical instability)
    max_logit_threshold: float = 1e4           # Warn threshold
    halt_on_exploding_logits: bool = True      # Stop if exceeded
    
    # Early stopping
    early_stopping_patience: int = 10          # Epochs without improvement
    early_stopping_metric: str = "val_loss"    # Metric to monitor
    early_stopping_mode: str = "min"           # "min" or "max"
    
    @classmethod
    def from_yaml(cls, path: Path) -> "StabilizationConfig":
        """Load config from YAML file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        
        stab = cfg.get("stabilization", {})
        return cls(
            lr=stab.get("lr", cls.lr),
            weight_decay=stab.get("weight_decay", cls.weight_decay),
            use_adamw=stab.get("use_adamw", cls.use_adamw),
            clip_grad_norm=stab.get("clip_grad_norm", cls.clip_grad_norm),
            init_std=stab.get("init_std", cls.init_std),
            init_bias_zero=stab.get("init_bias_zero", cls.init_bias_zero),
            disable_batchnorm_if_batchsize_lt=stab.get(
                "disable_batchnorm_if_batchsize_lt", 
                cls.disable_batchnorm_if_batchsize_lt
            ),
            use_layernorm=stab.get("use_layernorm", cls.use_layernorm),
            max_logit_threshold=stab.get("max_logit_threshold", cls.max_logit_threshold),
            halt_on_exploding_logits=stab.get(
                "halt_on_exploding_logits", 
                cls.halt_on_exploding_logits
            ),
            early_stopping_patience=stab.get(
                "early_stopping_patience", 
                cls.early_stopping_patience
            ),
            early_stopping_metric=stab.get(
                "early_stopping_metric", 
                cls.early_stopping_metric
            ),
            early_stopping_mode=stab.get(
                "early_stopping_mode", 
                cls.early_stopping_mode
            ),
        )


@dataclass
class EpochLog:
    """Per-epoch training log for stability monitoring."""
    epoch: int
    train_loss: float
    val_loss: float
    max_grad_norm: float
    max_logit: float
    learning_rate: float
    train_acc: float = 0.0
    val_auroc: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "max_grad_norm": self.max_grad_norm,
            "max_logit": self.max_logit,
            "learning_rate": self.learning_rate,
            "train_acc": self.train_acc,
            "val_auroc": self.val_auroc,
        }
    
    def __str__(self) -> str:
        return (
            f"Epoch {self.epoch:3d} | "
            f"train_loss: {self.train_loss:.4f} | "
            f"val_loss: {self.val_loss:.4f} | "
            f"max_grad: {self.max_grad_norm:.2e} | "
            f"max_logit: {self.max_logit:.2e} | "
            f"lr: {self.learning_rate:.2e} | "
            f"val_auroc: {self.val_auroc:.4f}"
        )


# ==============================================================================
# DATASET
# ==============================================================================

class CoralOmicsDataset(Dataset):
    """
    Dataset wrapper for coral multi-omics data compatible with LASER-VFL.
    
    Each omics block is treated as a "client" partition.
    Provides feature blocks + label + mask (all blocks present by default).
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
            omic_names: Ordered list of omics names (defines block order)
            sample_indices: Indices to use from the full dataset
            p_miss: Probability of missing features (for LASER training)
        """
        self.omic_names = omic_names
        self.n_clients = len(omic_names)
        self.p_miss = p_miss
        
        # Extract samples for this split
        self.labels = labels[sample_indices].astype(np.int64)
        self.features = {}
        for name in omic_names:
            self.features[name] = data[name][sample_indices].astype(np.float32)
        
        self.n_samples = len(sample_indices)
        
        # Precompute mask patterns (one per batch, simplified to one per sample here)
        self._generate_masks()
    
    def _generate_masks(self):
        """Generate availability masks for LASER-style missingness."""
        if self.p_miss == 0.0:
            # All blocks always available
            self.masks = [torch.ones(self.n_clients, dtype=torch.bool) 
                         for _ in range(self.n_samples)]
        else:
            # Bernoulli missingness per block
            self.masks = []
            for _ in range(self.n_samples):
                mask = np.random.rand(self.n_clients) > self.p_miss
                # Ensure at least one block is present
                if not mask.any():
                    mask[np.random.randint(self.n_clients)] = True
                self.masks.append(torch.tensor(mask, dtype=torch.bool))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Return features per block, label, and mask.
        
        Returns:
            (*features_per_block, label, mask)
        """
        features_per_block = tuple(
            torch.tensor(self.features[name][idx], dtype=torch.float32)
            for name in self.omic_names
        )
        label = self.labels[idx]
        mask = self.masks[idx]
        
        return (*features_per_block, label, mask)


def collate_fn(batch):
    """
    Custom collate function for LASER-VFL compatibility.
    
    Stacks features per block, labels, and uses first mask (batch-level).
    """
    n_blocks = len(batch[0]) - 2  # features, ..., label, mask
    
    features_per_block = [
        torch.stack([item[i] for item in batch])
        for i in range(n_blocks)
    ]
    labels = torch.tensor([item[-2] for item in batch], dtype=torch.long)
    mask = batch[0][-1]  # Use first sample's mask (batch-level pattern)
    
    return (*features_per_block, labels, mask)


# ==============================================================================
# STABILIZED MODEL COMPONENTS (WRAPPERS AROUND ORIGINAL LASER CLASSES)
# ==============================================================================
# 
# These classes wrap the original LASER model components with minimal 
# stabilization modifications required for the extreme P>>N coral data regime.
# The stabilization modifications are:
#   1. LayerNorm wrapper (optional) - added to MLPFeatureExtractor output
#   2. Small weight init - applied to FusionModel classifier
#   3. Logit monitoring - added to track numerical stability
#
# ARCHITECTURAL TRANSPARENCY:
# - MLPFeatureExtractor: IDENTICAL to original (Linear->ReLU->Linear->ReLU->Linear)
# - FusionModel: IDENTICAL to original (mean aggregation + Linear classifier)
#   with only weight initialization modified for stability
# - Task sampling: IDENTICAL to original LASER powerset-based sampling
# ==============================================================================

class StabilizedMLPFeatureExtractor(nn.Module):
    """
    Wrapper around original LASER MLPFeatureExtractor with optional stabilization.
    
    This wraps the ORIGINAL MLPFeatureExtractor from LASER-VFL-master/models/mlp.py
    and optionally adds LayerNorm output normalization for numerical stability
    in small-batch regimes.
    
    Architecture:
    - Core: Original MLPFeatureExtractor (Linear(in, hidden) -> ReLU -> 
            Linear(hidden, hidden) -> ReLU -> Linear(hidden, cut_dim))
    - Optional: LayerNorm on output (only when use_layernorm=True)
    - Optional: Dropout after core (only when dropout > 0)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_dim: int, 
        cut_dim: int,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Use the ORIGINAL LASER MLPFeatureExtractor as the core
        # This ensures architectural fidelity with vanilla LASER
        self.core_extractor = MLPFeatureExtractor(
            input_size=input_size,
            cut_dim=cut_dim,
            hidden_dim=hidden_dim
        )
        
        # STABILIZATION: Optional LayerNorm on output (for small-batch stability)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.output_norm = nn.LayerNorm(cut_dim)
        
        # STABILIZATION: Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        # Pass through original LASER feature extractor
        out = self.core_extractor(x)
        
        # Apply optional stabilization layers
        if self.use_layernorm:
            out = self.output_norm(out)
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out


class StabilizedFusionModel(nn.Module):
    """
    Wrapper around original LASER FusionModel with stabilized initialization.
    
    This uses the ORIGINAL FusionModel architecture from LASER-VFL-master/models/model_utils.py
    with only the weight initialization modified for stability in small-N regimes.
    
    Architecture:
    - Core: Original FusionModel (mean/sum/concat aggregation + Linear classifier)
    - Modification: Small weight init (std=1e-3) to prevent large initial logits
    
    The small weight initialization is critical for preventing softmax saturation
    on small datasets where large initial logits can lead to gradient vanishing.
    """
    
    def __init__(
        self, 
        cut_dim: int, 
        num_classes: int, 
        aggregation: str = "mean",
        num_clients: int = None,
        init_std: float = 1e-3,
        init_bias_zero: bool = True,
    ):
        super().__init__()
        
        # Use the ORIGINAL LASER FusionModel as the core
        # This ensures architectural fidelity with vanilla LASER
        self.core_fusion = FusionModel(
            cut_dim=cut_dim,
            num_classes=num_classes,
            aggregation=aggregation,
            num_clients=num_clients
        )
        
        # STABILIZATION: Apply small weight init to the classifier
        # This is the ONLY modification to vanilla LASER FusionModel
        nn.init.normal_(self.core_fusion.classifier.weight, mean=0.0, std=init_std)
        if init_bias_zero:
            nn.init.zeros_(self.core_fusion.classifier.bias)
    
    def forward(self, x):
        # Pass through original LASER FusionModel
        return self.core_fusion(x)


# ==============================================================================
# STABILIZED LASER MODEL
# ==============================================================================

class StabilizedLASERModel(nn.Module):
    """
    LASER-VFL model with minimal stabilization for small-N, high-P data.
    
    This model implements the ORIGINAL LASER architecture from BaseLaserModel
    (LASER-VFL-master/models/model_utils.py) with:
    
    CORE ARCHITECTURE (IDENTICAL TO VANILLA LASER):
    - Feature extractors: One MLPFeatureExtractor per client (omics type)
    - Fusion models: One FusionModel per client head
    - Task sampling: Powerset-based sampling (each head samples one task per subset size)
    - Forward pass: Same as BaseLaserModel with observed_blocks support
    
    STABILIZATION MODIFICATIONS (ONLY):
    - LayerNorm wrapper on feature extractor outputs (optional, via StabilizedMLPFeatureExtractor)
    - Small weight initialization on fusion heads (via StabilizedFusionModel)
    - Logit monitoring hooks for numerical stability detection
    
    These modifications do NOT change the LASER learning algorithm, only address
    numerical instability in the extreme P>>N regime (N=13, P=90,000).
    """
    
    def __init__(
        self, 
        feature_extractors: nn.ModuleList,
        fusion_models: nn.ModuleList,
        num_clients: int,
        omic_names: List[str],
    ):
        super().__init__()
        self.num_clients = num_clients
        self.omic_names = omic_names
        self.feature_extractors = feature_extractors
        self.fusion_models = fusion_models
        
        # Precompute powerset for LASER's task-sampling
        self.powerset = self._powerset_except_empty(num_clients)
        
        # For logit monitoring
        self._max_logit = 0.0
    
    def _powerset_except_empty(self, n: int) -> List[Tuple[int, ...]]:
        """Generate all non-empty subsets of {0, 1, ..., n-1}."""
        from itertools import chain, combinations
        return list(chain.from_iterable(
            combinations(range(n), r) for r in range(1, n + 1)
        ))
    
    def get_block(self, x, i):
        """x is tuple of (block_0, block_1, ..., block_n)"""
        return x[i]
    
    def forward(
        self, 
        x: Tuple[torch.Tensor, ...], 
        training: bool = True, 
        observed_blocks: List[int] = None
    ) -> List[Dict[Tuple[int, ...], torch.Tensor]]:
        """
        Forward pass with LASER's task-sampling.
        
        Args:
            x: Tuple of feature tensors, one per omics
            training: Whether in training mode (uses task sampling)
            observed_blocks: List of available block indices
            
        Returns:
            List of dicts mapping client subsets to logit tensors
        """
        import random
        
        if observed_blocks is None:
            observed_blocks = list(range(self.num_clients))
        
        # Compute embeddings for observed blocks
        embeddings = {}
        for i in observed_blocks:
            local_input = self.get_block(x, i)
            embeddings[i] = self.feature_extractors[i](local_input)
        
        if training:
            # LASER's task sampling: each head samples one task per subset size
            outputs = []
            for i, fusion_model in enumerate(self.fusion_models):
                if i not in observed_blocks:
                    continue
                    
                sets_considered_by_head = [
                    clients_l for clients_l in self.powerset 
                    if i in clients_l and set(clients_l).issubset(set(observed_blocks))
                ]
                
                head_output = {}
                for num_clients_in_agg in range(1, len(observed_blocks) + 1):
                    set_to_sample = [
                        client_set for client_set in sets_considered_by_head 
                        if len(client_set) == num_clients_in_agg
                    ]
                    if set_to_sample:
                        [sample] = random.sample(set_to_sample, 1)
                        logits = fusion_model([embeddings[j] for j in sample])
                        
                        # Monitor max logit for stability
                        self._max_logit = max(
                            self._max_logit, 
                            logits.abs().max().item()
                        )
                        
                        head_output[sample] = logits
                        
                outputs.append(head_output)
        else:
            # Evaluation: compute all subset predictions
            outputs = []
            for i, fusion_model in enumerate(self.fusion_models):
                head_output = {}
                for clients_l in self.powerset:
                    if i in clients_l:
                        logits = fusion_model([embeddings[j] for j in clients_l])
                        self._max_logit = max(
                            self._max_logit, 
                            logits.abs().max().item()
                        )
                        head_output[clients_l] = logits
                outputs.append(head_output)
        
        return outputs
    
    def get_max_logit(self) -> float:
        """Get maximum absolute logit observed since last reset."""
        return self._max_logit
    
    def reset_max_logit(self):
        """Reset max logit tracker."""
        self._max_logit = 0.0


# ==============================================================================
# LASER ADAPTER
# ==============================================================================

class LASERAdapter:
    """
    Stabilized adapter to run LASER-VFL on coral multi-omics data.
    
    Provides a unified interface for the benchmark runner with full
    instrumentation for stability monitoring and logging.
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        labels: np.ndarray,
        omic_names: List[str],
        config: Dict[str, Any],
        device: torch.device = None,
        stabilization_config: StabilizationConfig = None,
    ):
        """
        Initialize adapter.
        
        Args:
            data: Dict of omics data arrays (will be scaled internally)
            labels: Label array
            omic_names: Ordered list of omics names
            config: Configuration dict with hyperparameters
            device: Torch device
            stabilization_config: Stabilization settings (uses defaults if None)
        """
        self.raw_data = data
        self.labels = labels
        self.omic_names = omic_names
        self.n_clients = len(omic_names)
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load stabilization config
        self.stab = stabilization_config or StabilizationConfig()
        
        # Override config with stabilization settings if not explicitly set
        if "lr" not in config:
            config["lr"] = self.stab.lr
        if "weight_decay" not in config:
            config["weight_decay"] = self.stab.weight_decay
        
        # Scalers will be fit on training data
        self.scalers: Dict[str, Dict[str, np.ndarray]] = {}
        self.scaled_data: Dict[str, np.ndarray] = {}
        
        # Model and training state
        self.model: Optional[StabilizedLASERModel] = None
        self.optimizer = None
        self.criterion = None
        
        # Training logs
        self.epoch_logs: List[EpochLog] = []
        self.log_file: Optional[Path] = None
        
    def _fit_scalers(self, train_indices: np.ndarray):
        """
        Fit StandardScaler on training data only (prevents data leakage).
        
        Args:
            train_indices: Indices of training samples
        """
        self.scalers = {}
        for name, arr in self.raw_data.items():
            train_data = arr[train_indices]
            mean = train_data.mean(axis=0, keepdims=True)
            std = train_data.std(axis=0, keepdims=True) + 1e-8  # Avoid div by zero
            self.scalers[name] = {"mean": mean, "std": std}
    
    def _transform_data(self) -> Dict[str, np.ndarray]:
        """
        Apply fitted scalers to all data.
        
        Returns:
            Dict of scaled arrays
        """
        self.scaled_data = {}
        for name, arr in self.raw_data.items():
            scaler = self.scalers[name]
            self.scaled_data[name] = (
                (arr - scaler["mean"]) / scaler["std"]
            ).astype(np.float32)
        return self.scaled_data
        
    def _get_feature_dims(self) -> List[int]:
        """Get feature dimension for each omics block."""
        return [self.raw_data[name].shape[1] for name in self.omic_names]
    
    def _create_model(self):
        """
        Create stabilized LASER-VFL model for coral data.
        
        Uses:
        - LayerNorm instead of BatchNorm
        - Small weight init for classification heads
        """
        feature_dims = self._get_feature_dims()
        hidden_dim = self.config.get("hidden_dim", 128)
        cut_dim = self.config.get("embedding_dim", 64)
        dropout = self.config.get("dropout", 0.0)
        n_classes = 2  # Binary with CrossEntropyLoss
        
        # Create stabilized feature extractors (one per omics)
        feature_extractors = nn.ModuleList()
        for dim in feature_dims:
            extractor = StabilizedMLPFeatureExtractor(
                input_size=dim,
                hidden_dim=hidden_dim,
                cut_dim=cut_dim,
                use_layernorm=self.stab.use_layernorm,
                dropout=dropout,
            )
            feature_extractors.append(extractor)
        
        # Create stabilized fusion models (one per client head)
        fusion_models = nn.ModuleList([
            StabilizedFusionModel(
                cut_dim=cut_dim,
                num_classes=n_classes,
                init_std=self.stab.init_std,
                init_bias_zero=self.stab.init_bias_zero,
            )
            for _ in range(self.n_clients)
        ])
        
        self.model = StabilizedLASERModel(
            feature_extractors, fusion_models,
            self.n_clients, self.omic_names
        ).to(self.device)
        
        return self.model
    
    def _create_optimizer(self):
        """
        Create optimizer with stabilization settings.
        
        Uses AdamW with lower LR and weight decay by default.
        """
        lr = self.config.get("lr", self.stab.lr)
        weight_decay = self.config.get("weight_decay", self.stab.weight_decay)
        
        if self.stab.use_adamw:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        
        return self.optimizer
    
    def _create_criterion(self):
        """Create loss criterion."""
        self.criterion = nn.CrossEntropyLoss()
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
        
        Args:
            sample_indices: Array of sample indices to include
            batch_size: Batch size
            shuffle: Whether to shuffle
            p_miss: Missing probability for LASER training
            drop_last: Drop last incomplete batch
        
        Returns:
            DataLoader
        """
        dataset = CoralOmicsDataset(
            data=self.scaled_data,
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
    
    def train_epoch(
        self, 
        train_loader: DataLoader
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Train for one epoch with stability monitoring.
        
        Returns:
            Tuple of (metrics_dict, max_grad_norm, max_logit)
        """
        self.model.train()
        self.model.reset_max_logit()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        max_grad_norm = 0.0
        
        for batch in train_loader:
            *inputs, targets, mask = batch
            inputs = [x.to(self.device) for x in inputs]
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get observed blocks from mask
            observed_blocks = torch.nonzero(mask).view(-1).tolist()
            if not observed_blocks:
                continue
            
            # Forward pass with LASER's task-sampling
            outputs_per_head = self.model(
                tuple(inputs), 
                training=True, 
                observed_blocks=observed_blocks
            )
            
            # Compute loss over all heads and tasks (LASER's loss weighting)
            batch_loss = 0.0
            for head_outputs in outputs_per_head:
                head_loss = 0.0
                for clients_subset, outputs in head_outputs.items():
                    loss = self.criterion(outputs, targets)
                    
                    # LASER normalization constant
                    norm_constant = 1.0 / len(clients_subset)
                    n, k = (len(observed_blocks) - 1, len(clients_subset) - 1)
                    norm_constant *= math.comb(n, k)
                    
                    head_loss += loss * norm_constant
                    
                batch_loss += head_loss
            
            batch_loss.backward()
            
            # STABILIZATION: Gradient clipping before optimizer step
            # This prevents exploding gradients on small batches
            if self.stab.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.stab.clip_grad_norm,
                )
                max_grad_norm = max(max_grad_norm, grad_norm.item())
            
            self.optimizer.step()
            
            total_loss += batch_loss.item() * len(targets)
            
            # Use first head's full-set prediction for accuracy
            full_set = tuple(observed_blocks)
            if full_set in outputs_per_head[0]:
                preds = outputs_per_head[0][full_set].argmax(dim=1)
                total_correct += (preds == targets).sum().item()
            total_samples += len(targets)
        
        max_logit = self.model.get_max_logit()
        
        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }, max_grad_norm, max_logit
    
    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test DataLoader
            return_predictions: Whether to return raw predictions
        
        Returns:
            Dict with metrics (and optionally predictions)
        """
        self.model.eval()
        self.model.reset_max_logit()
        
        all_probs = []
        all_targets = []
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                *inputs, targets, mask = batch
                inputs = [x.to(self.device) for x in inputs]
                targets = targets.to(self.device)
                
                observed_blocks = torch.nonzero(mask).view(-1).tolist()
                if not observed_blocks:
                    continue
                
                # Forward (testing mode)
                outputs_per_head = self.model(
                    tuple(inputs),
                    training=False,
                    observed_blocks=observed_blocks
                )
                
                # Use first head's full-set prediction
                full_set = tuple(observed_blocks)
                if full_set in outputs_per_head[0]:
                    logits = outputs_per_head[0][full_set]
                    loss = self.criterion(logits, targets)
                    
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    
                    total_loss += loss.item() * len(targets)
                    total_correct += (preds == targets).sum().item()
                    
                    all_probs.append(probs[:, 1].cpu().numpy())  # P(class=1)
                    all_targets.append(targets.cpu().numpy())
                
                total_samples += len(targets)
        
        # Compute metrics
        all_probs = np.concatenate(all_probs) if all_probs else np.array([])
        all_targets = np.concatenate(all_targets) if all_targets else np.array([])
        
        results = {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "max_logit": self.model.get_max_logit(),
        }
        
        if len(all_probs) > 0:
            from sklearn.metrics import (
                roc_auc_score, average_precision_score, 
                f1_score, brier_score_loss
            )
            
            preds_binary = (all_probs >= 0.5).astype(int)
            
            # Guard against single-class val sets (e.g., LOOCV with 1 sample).
            # sklearn >=1.5 returns nan instead of raising ValueError when
            # only one class is present, so we check explicitly first.
            if len(np.unique(all_targets)) > 1:
                try:
                    auroc = float(roc_auc_score(all_targets, all_probs))
                    results["auroc"] = auroc if not np.isnan(auroc) else 0.5
                except ValueError:
                    results["auroc"] = 0.5
            else:
                results["auroc"] = 0.5
            
            if len(np.unique(all_targets)) > 1:
                try:
                    auprc = float(average_precision_score(all_targets, all_probs))
                    results["auprc"] = auprc if not np.isnan(auprc) else float(np.mean(all_targets))
                except ValueError:
                    results["auprc"] = float(np.mean(all_targets))
            else:
                results["auprc"] = float(np.mean(all_targets))
            
            results["f1"] = float(f1_score(all_targets, preds_binary, zero_division=0))
            results["brier"] = float(brier_score_loss(all_targets, all_probs))
        
        if return_predictions:
            results["predictions"] = all_probs
            results["targets"] = all_targets
        
        return results
    
    def _log_epoch(self, log: EpochLog):
        """Log epoch to file and list."""
        self.epoch_logs.append(log)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(str(log) + "\n")
        
        logger.info(str(log))
    
    def _check_stability(self, max_logit: float, epoch: int) -> bool:
        """
        Check for numerical instability.
        
        Returns:
            True if stable, False if should halt
        """
        if max_logit > self.stab.max_logit_threshold:
            logger.warning(
                f"STABILITY WARNING: max_logit={max_logit:.2e} exceeds "
                f"threshold={self.stab.max_logit_threshold:.2e} at epoch {epoch}"
            )
            if self.stab.halt_on_exploding_logits:
                logger.error("Halting training due to exploding logits")
                return False
        return True
    
    def train(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = None,
        p_miss_train: float = 0.0,
        log_file: Path = None,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping and stability monitoring.
        
        Args:
            train_indices: Training sample indices
            val_indices: Validation sample indices
            n_epochs: Maximum epochs
            batch_size: Batch size
            early_stopping_patience: Epochs without improvement before stopping
            p_miss_train: Missing probability during training
            log_file: Path to write per-epoch logs
        
        Returns:
            Training history with stability metrics
        """
        # Use stabilization config patience if not specified
        patience = early_stopping_patience or self.config.get("patience", self.stab.early_stopping_patience)
        # FIX: was using StabilizationConfig default (10); now reads from benchmark config dict (15)
        
        # Setup logging
        self.log_file = log_file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                f.write("# LASER-VFL Stabilized Training Log\n")
                f.write(f"# Stabilization config: lr={self.stab.lr}, ")
                f.write(f"clip_norm={self.stab.clip_grad_norm}, ")
                f.write(f"init_std={self.stab.init_std}\n")
                f.write("# " + "="*70 + "\n")
        
        self.epoch_logs = []
        
        # Fit scalers on training data only
        self._fit_scalers(train_indices)
        self._transform_data()
        
        # Create model and optimizer
        self._create_model()
        self._create_optimizer()
        self._create_criterion()
        
        # Determine if we should drop last batch for small datasets
        drop_last = len(train_indices) > batch_size
        
        train_loader = self.create_dataloader(
            train_indices, batch_size, shuffle=True, 
            p_miss=p_miss_train, drop_last=drop_last
        )
        val_loader = self.create_dataloader(
            val_indices, batch_size, shuffle=False, p_miss=0.0
        )
        
        # Early stopping state
        best_val_metric = float("inf") if self.stab.early_stopping_mode == "min" else -float("inf")
        best_model_state = None
        patience_counter = 0
        history = {"train": [], "val": [], "epoch_logs": []}
        
        for epoch in range(n_epochs):
            # Train epoch
            train_metrics, max_grad_norm, max_logit = self.train_epoch(train_loader)
            
            # Check stability
            if not self._check_stability(max_logit, epoch):
                history["halted_due_to_instability"] = True
                break
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log epoch
            log = EpochLog(
                epoch=epoch + 1,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                max_grad_norm=max_grad_norm,
                max_logit=max_logit,
                learning_rate=current_lr,
                train_acc=train_metrics["accuracy"],
                val_auroc=val_metrics.get("auroc", 0.5),
            )
            self._log_epoch(log)
            
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)
            history["epoch_logs"].append(log.to_dict())
            
            # Check for NaN
            if np.isnan(train_metrics["loss"]) or np.isnan(val_metrics["loss"]):
                logger.error(f"NaN loss detected at epoch {epoch+1}")
                history["halted_due_to_nan"] = True
                break
            
            # Early stopping check
            if self.stab.early_stopping_metric == "val_loss":
                current_metric = val_metrics["loss"]
            else:
                current_metric = val_metrics.get("auroc", 0.5)
            
            is_better = (
                (self.stab.early_stopping_mode == "min" and current_metric < best_val_metric) or
                (self.stab.early_stopping_mode == "max" and current_metric > best_val_metric)
            )
            
            if is_better:
                best_val_metric = current_metric
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Compute final best val auroc
        final_val = self.evaluate(val_loader)
        
        history["best_val_auroc"] = final_val.get("auroc", 0.5)
        history["best_epoch"] = len(history["val"]) - patience_counter
        history["total_epochs"] = len(history["val"])
        history["stabilization_config"] = {
            "lr": self.stab.lr,
            "weight_decay": self.stab.weight_decay,
            "clip_grad_norm": self.stab.clip_grad_norm,
            "init_std": self.stab.init_std,
            "use_layernorm": self.stab.use_layernorm,
            "early_stopping_patience": patience,
        }
        
        return history
    
    def predict(
        self,
        test_indices: np.ndarray,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for test samples.
        
        Args:
            test_indices: Test sample indices
            batch_size: Batch size
        
        Returns:
            (predictions, targets) arrays
        """
        test_loader = self.create_dataloader(
            test_indices, batch_size, shuffle=False, p_miss=0.0
        )
        results = self.evaluate(test_loader, return_predictions=True)
        return results["predictions"], results["targets"]
    
    def get_epoch_logs(self) -> List[Dict[str, Any]]:
        """Get all epoch logs as list of dicts."""
        return [log.to_dict() for log in self.epoch_logs]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_coral_data(data_dir: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load coral multi-omics data in canonical format.
    
    Data format: rows=samples, first column='sample_id', rest are features.
    
    Returns:
        data: Dict mapping omic name -> array (n_samples, n_features)
        sample_ids: List of sample identifiers
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
    import json
    
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    print("Loading data...")
    data, sample_ids = load_coral_data(data_dir)
    print(f"Loaded {len(data)} omics for {len(sample_ids)} samples")
    for name, arr in data.items():
        print(f"  {name}: {arr.shape}")
    
    # Infer labels
    labels = np.array([1 if "HiT" in s else 0 for s in sample_ids])
    print(f"Labels: {np.bincount(labels)}")
    
    # Test adapter with stabilization
    config = {
        "hidden_dim": 64,
        "embedding_dim": 32,
    }
    
    stab_config = StabilizationConfig()
    
    adapter = LASERAdapter(
        data=data,
        labels=labels,
        omic_names=list(data.keys()),
        config=config,
        stabilization_config=stab_config,
    )
    
    # Quick train test
    train_idx = np.arange(min(8, len(sample_ids) - 3))
    val_idx = np.arange(min(8, len(sample_ids) - 3), len(sample_ids))
    
    log_path = PROJECT_ROOT / "benchmarks" / "logs" / "test_laser_train.log"
    
    print(f"\nTraining stabilized LASER adapter...")
    print(f"Stabilization: lr={stab_config.lr}, clip_norm={stab_config.clip_grad_norm}")
    
    history = adapter.train(
        train_idx, val_idx,
        n_epochs=10,
        batch_size=4,
        log_file=log_path,
    )
    
    print(f"\nBest val AUROC: {history['best_val_auroc']:.4f}")
    print(f"Total epochs: {history['total_epochs']}")
    print(f"\nEpoch logs saved to: {log_path}")
