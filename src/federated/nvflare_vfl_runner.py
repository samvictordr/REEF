#!/usr/bin/env python3
"""
NVFlare Vertical Federated Learning Runner for Coral Multi-Omics.

This module provides a complete VFL pipeline that:
1. Simulates NVFlare client-server architecture locally for development
2. Can be deployed to real NVFlare infrastructure for production
3. Integrates domain saliency+XAI for feature importance analysis

Architecture:
    Client Silo A (Transcriptomics) --> [OmicsEncoder] --> Embedding
    Client Silo B (Proteomics)      --> [OmicsEncoder] --> Embedding
    Client Silo C (Metabolomics)    --> [OmicsEncoder] --> Embedding  --> Server [FusionHead] --> Prediction
    Client Silo D (Microbiome)      --> [OmicsEncoder] --> Embedding

Key Features:
- True VFL: Each silo only sees its own features, never raw data from others
- Split Learning: Gradients flow back from server to clients
- Domain Saliency Integration: Feature importance computed per-omic
- Mosaic Masking: Handles missing omics gracefully
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VFLConfig:
    """Configuration for VFL training."""
    n_rounds: int = 100
    batch_size: int = 8
    lr_client: float = 5e-5  # Lower LR for stability with high-dim data
    lr_server: float = 5e-4
    weight_decay: float = 1e-4
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.3
    patience: int = 15
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"


@dataclass
class OmicSilo:
    """Represents a single omics data silo (client)."""
    name: str
    features: np.ndarray  # (n_samples, n_features)
    scaler: StandardScaler = None
    encoder: nn.Module = None
    optimizer: torch.optim.Optimizer = None
    
    def __post_init__(self):
        self.n_features = self.features.shape[1] if self.features is not None else 0
        self.scaler = StandardScaler()


# =============================================================================
# Neural Network Components (Mirror NVFlare job components)
# =============================================================================

class OmicsEncoder(nn.Module):
    """
    Local encoder for a single omics type.
    
    This is the model that runs on each client silo.
    Only the client sees the raw features - only embeddings are sent to server.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Using LayerNorm for stability with small batches
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VFLFusionHead(nn.Module):
    """
    Server-side fusion head that combines embeddings from all silos.
    
    Supports:
    - Concatenation fusion (default)
    - Attention-based fusion
    - Missing omics via masking
    """
    
    def __init__(
        self,
        n_silos: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 1,
        fusion_method: str = "concat",
    ):
        super().__init__()
        
        self.n_silos = n_silos
        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "concat":
            input_dim = n_silos * embedding_dim
        elif fusion_method == "attention":
            input_dim = embedding_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=4,
                batch_first=True,
            )
        else:  # mean
            input_dim = embedding_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes),
        )
    
    def forward(
        self, 
        embeddings: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: List of (batch, embedding_dim) tensors from each silo
            mask: (batch, n_silos) availability mask. 1=available, 0=missing
        
        Returns:
            Logits (batch, n_classes)
        """
        stacked = torch.stack(embeddings, dim=1)  # (B, n_silos, embedding_dim)
        
        if mask is not None:
            # Zero out missing silos
            mask_expanded = mask.unsqueeze(-1)  # (B, n_silos, 1)
            stacked = stacked * mask_expanded
        
        if self.fusion_method == "concat":
            fused = stacked.view(stacked.shape[0], -1)
        elif self.fusion_method == "attention":
            fused, _ = self.attention(stacked, stacked, stacked)
            fused = fused.mean(dim=1)
        else:
            fused = stacked.mean(dim=1)
        
        return self.classifier(fused)


# =============================================================================
# VFL Trainer (Simulates NVFlare Client-Server locally)
# =============================================================================

class NVFlareVFLTrainer:
    """
    Vertical Federated Learning trainer that simulates NVFlare architecture.
    
    This trainer:
    1. Manages client silos (each with own encoder)
    2. Manages server (fusion head + classifier)
    3. Implements split learning gradient flow
    4. Provides domain saliency integration hooks
    
    Can be adapted to run on real NVFlare by replacing the simulation
    with actual NVFlare executor/controller calls.
    """
    
    def __init__(
        self,
        silos: Dict[str, OmicSilo],
        labels: np.ndarray,
        config: VFLConfig,
    ):
        self.silos = silos
        self.omic_names = list(silos.keys())
        self.n_silos = len(silos)
        self.labels = labels
        self.config = config
        
        # Device setup
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize client encoders
        self._init_clients()
        
        # Initialize server fusion head
        self._init_server()
        
        # Training state
        self.best_val_auroc = 0.0
        self.best_state = None
        self.patience_counter = 0
    
    def _init_clients(self):
        """Initialize encoder for each client silo."""
        for name, silo in self.silos.items():
            silo.encoder = OmicsEncoder(
                input_dim=silo.n_features,
                hidden_dim=self.config.hidden_dim,
                embedding_dim=self.config.embedding_dim,
                dropout=self.config.dropout,
            ).to(self.device)
            
            silo.optimizer = torch.optim.AdamW(
                silo.encoder.parameters(),
                lr=self.config.lr_client,
                weight_decay=self.config.weight_decay,
            )
            
            logger.info(f"Initialized silo '{name}': {silo.n_features} features")
    
    def _init_server(self):
        """Initialize server fusion head."""
        self.fusion_head = VFLFusionHead(
            n_silos=self.n_silos,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            n_classes=1,  # Binary classification
        ).to(self.device)
        
        self.server_optimizer = torch.optim.AdamW(
            self.fusion_head.parameters(),
            lr=self.config.lr_server,
            weight_decay=self.config.weight_decay,
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def _prepare_batch(
        self, 
        indices: np.ndarray,
        train_indices: np.ndarray,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Prepare a batch of data.
        
        Args:
            indices: Sample indices for this batch
            train_indices: All training indices (for scaler fitting)
        
        Returns:
            features_dict: Dict[omic_name -> (batch, n_features) tensor]
            labels: (batch,) tensor
        """
        features_dict = {}
        
        for name, silo in self.silos.items():
            # Scale using training data statistics
            if not hasattr(silo.scaler, 'mean_') or silo.scaler.mean_ is None:
                silo.scaler.fit(silo.features[train_indices])
            
            batch_data = silo.scaler.transform(silo.features[indices])
            
            # Clip extreme values for numerical stability
            batch_data = np.clip(batch_data, -5, 5)
            
            # Replace any NaN/Inf with 0
            batch_data = np.nan_to_num(batch_data, nan=0.0, posinf=5.0, neginf=-5.0)
            
            features_dict[name] = torch.tensor(
                batch_data, dtype=torch.float32, device=self.device
            )
        
        batch_labels = torch.tensor(
            self.labels[indices], dtype=torch.float32, device=self.device
        )
        
        return features_dict, batch_labels
    
    def _client_forward(
        self, 
        features_dict: Dict[str, torch.Tensor],
        retain_grads: bool = True,
    ) -> List[torch.Tensor]:
        """
        Client forward pass: each silo computes its embedding.
        
        In real NVFlare, this would happen on each client machine.
        Only embeddings are sent to server, never raw features.
        """
        embeddings = []
        for name in self.omic_names:
            silo = self.silos[name]
            emb = silo.encoder(features_dict[name])
            if retain_grads and emb.requires_grad:
                emb.retain_grad()  # For backward pass
            embeddings.append(emb)
        return embeddings
    
    def _server_forward(
        self, 
        embeddings: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Server forward pass: fuse embeddings and classify.
        
        In real NVFlare, server receives embeddings, computes loss,
        and sends gradients back to clients.
        """
        return self.fusion_head(embeddings, mask)
    
    def _backward_and_update(
        self, 
        loss: torch.Tensor,
        embeddings: List[torch.Tensor],
    ):
        """
        Backward pass: compute gradients and update models.
        
        In real NVFlare:
        1. Server computes loss.backward() on its fusion head
        2. Server sends embedding gradients to each client
        3. Each client continues backward through its encoder
        """
        # Zero all gradients
        self.server_optimizer.zero_grad()
        for silo in self.silos.values():
            silo.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.fusion_head.parameters(), self.config.grad_clip
        )
        for silo in self.silos.values():
            torch.nn.utils.clip_grad_norm_(
                silo.encoder.parameters(), self.config.grad_clip
            )
        
        # Update server
        self.server_optimizer.step()
        
        # Update clients (in real NVFlare, gradients would be sent to clients)
        for silo in self.silos.values():
            silo.optimizer.step()
    
    def train_round(
        self, 
        train_indices: np.ndarray,
    ) -> float:
        """
        Execute one training round.
        
        Returns:
            Average loss for this round
        """
        # Set training mode
        self.fusion_head.train()
        for silo in self.silos.values():
            silo.encoder.train()
        
        # Mini-batch training
        n_samples = len(train_indices)
        batch_size = min(self.config.batch_size, n_samples)
        
        # Shuffle indices
        shuffled = np.random.permutation(train_indices)
        
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = shuffled[i:i+batch_size]
            if len(batch_indices) < 2:
                continue
            
            features_dict, labels = self._prepare_batch(batch_indices, train_indices)
            
            # Client forward
            embeddings = self._client_forward(features_dict)
            
            # Server forward
            logits = self._server_forward(embeddings)
            
            # Compute loss
            loss = self.criterion(logits.squeeze(), labels)
            
            # Skip if loss is NaN or Inf (stability check)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping batch due to invalid loss: {loss.item()}")
                continue
            
            # Backward and update
            self._backward_and_update(loss, embeddings)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def evaluate(
        self, 
        indices: np.ndarray,
        train_indices: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate on given indices.
        
        Returns:
            Dict with auroc, accuracy, loss
        """
        # Set eval mode
        self.fusion_head.eval()
        for silo in self.silos.values():
            silo.encoder.eval()
        
        features_dict, labels = self._prepare_batch(indices, train_indices)
        
        # Forward pass (no grads needed for evaluation)
        embeddings = self._client_forward(features_dict, retain_grads=False)
        logits = self._server_forward(embeddings)
        
        # Metrics
        loss = self.criterion(logits.squeeze(), labels).item()
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Handle NaN probabilities
        if np.any(np.isnan(probs)):
            logger.warning("NaN probabilities detected in evaluation")
            probs = np.nan_to_num(probs, nan=0.5)
        
        # Handle edge cases
        if len(np.unique(labels_np)) < 2:
            auroc = 0.5
        else:
            auroc = roc_auc_score(labels_np, probs)
        
        preds = (probs > 0.5).astype(int)
        accuracy = accuracy_score(labels_np, preds)
        
        return {"auroc": auroc, "accuracy": accuracy, "loss": loss}
    
    def train(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history
        """
        logger.info(f"Starting VFL training: {len(train_indices)} train, {len(val_indices)} val")
        
        history = {"train_loss": [], "val_auroc": [], "val_loss": []}
        
        for round_num in range(self.config.n_rounds):
            # Train
            train_loss = self.train_round(train_indices)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(val_indices, train_indices)
            history["val_auroc"].append(val_metrics["auroc"])
            history["val_loss"].append(val_metrics["loss"])
            
            # Early stopping
            if val_metrics["auroc"] > self.best_val_auroc:
                self.best_val_auroc = val_metrics["auroc"]
                self.best_state = self._get_state()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (round_num + 1) % 10 == 0 or round_num == 0:
                logger.info(
                    f"Round {round_num+1}/{self.config.n_rounds} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val AUROC: {val_metrics['auroc']:.4f} | "
                    f"Best: {self.best_val_auroc:.4f}"
                )
            
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at round {round_num+1}")
                break
        
        # Restore best model
        if self.best_state:
            self._load_state(self.best_state)
        
        return history
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current model state."""
        state = {
            "fusion_head": self.fusion_head.state_dict(),
            "encoders": {
                name: silo.encoder.state_dict() 
                for name, silo in self.silos.items()
            },
        }
        return state
    
    def _load_state(self, state: Dict[str, Any]):
        """Load model state."""
        self.fusion_head.load_state_dict(state["fusion_head"])
        for name, silo in self.silos.items():
            silo.encoder.load_state_dict(state["encoders"][name])
    
    def save_checkpoint(self, path: Path):
        """Save full checkpoint."""
        state = self._get_state()
        state["config"] = self.config.__dict__
        state["omic_names"] = self.omic_names
        state["omic_dims"] = {name: silo.n_features for name, silo in self.silos.items()}
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        state = torch.load(path, map_location=self.device)
        self._load_state(state)
        logger.info(f"Loaded checkpoint from {path}")
    
    def get_embeddings(
        self, 
        indices: np.ndarray,
        train_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Get embeddings for domain saliency analysis."""
        self.fusion_head.eval()
        for silo in self.silos.values():
            silo.encoder.eval()
        
        features_dict, _ = self._prepare_batch(indices, train_indices)
        
        embeddings_dict = {}
        with torch.no_grad():
            for name in self.omic_names:
                silo = self.silos[name]
                emb = silo.encoder(features_dict[name])
                embeddings_dict[name] = emb.cpu().numpy()
        
        return embeddings_dict


# =============================================================================
# Domain Saliency Integration
# =============================================================================

class VFLDomainWrapper(nn.Module):
    """
    Wrapper for domain saliency analysis of a single omic encoder.
    
    This allows computing feature importance for one omic at a time,
    while other omics are held constant (mean embedding).
    """
    
    def __init__(
        self,
        trainer: NVFlareVFLTrainer,
        target_omic: str,
        other_embeddings: Dict[str, np.ndarray],
    ):
        super().__init__()
        self.trainer = trainer
        self.target_omic = target_omic
        self.target_idx = trainer.omic_names.index(target_omic)
        
        # Store mean embeddings for other omics
        self.other_embeddings = {}
        for name, emb in other_embeddings.items():
            if name != target_omic:
                self.other_embeddings[name] = torch.tensor(
                    emb.mean(axis=0), dtype=torch.float32, device=trainer.device
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for domain saliency.
        
        Args:
            x: (batch, n_features) raw features for target omic
        
        Returns:
            (batch, 1) prediction probabilities
        """
        batch_size = x.shape[0]
        
        # Get target omic embedding
        silo = self.trainer.silos[self.target_omic]
        target_emb = silo.encoder(x)
        
        # Build full embedding list
        embeddings = []
        for name in self.trainer.omic_names:
            if name == self.target_omic:
                embeddings.append(target_emb)
            else:
                # Use mean embedding for other omics
                mean_emb = self.other_embeddings[name].unsqueeze(0).expand(batch_size, -1)
                embeddings.append(mean_emb)
        
        # Get prediction
        logits = self.trainer.fusion_head(embeddings)
        # Clone output to prevent inplace modification errors with SHAP's backward hooks
        return torch.sigmoid(logits).clone()


def compute_domain_saliency(
    trainer: NVFlareVFLTrainer,
    train_indices: np.ndarray,
    n_background: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute feature importance for each omic using interpretable XAI methods.
    
    Uses SHAP GradientExplainer as primary method with Integrated Gradients
    as fallback. Both methods provide theoretically-grounded feature attributions.
    
    Returns:
        Dict[omic_name -> (n_features,) importance array]
    """
    try:
        import shap
        has_shap = True
    except ImportError:
        logger.warning("SHAP not installed, using Integrated Gradients only")
        has_shap = False
    
    logger.info("Computing feature importance via interpretable XAI methods...")
    
    # Get embeddings for background
    embeddings = trainer.get_embeddings(train_indices, train_indices)
    
    importance = {}
    
    for omic_name in trainer.omic_names:
        logger.info(f"  Computing importance for {omic_name}...")
        
        silo = trainer.silos[omic_name]
        
        # Prepare background data
        bg_indices = np.random.choice(
            train_indices, size=min(n_background, len(train_indices)), replace=False
        )
        bg_data = silo.scaler.transform(silo.features[bg_indices])
        bg_tensor = torch.tensor(bg_data, dtype=torch.float32, device=trainer.device).clone()
        
        # Prepare test data
        test_data = silo.scaler.transform(silo.features[train_indices])
        test_tensor = torch.tensor(test_data, dtype=torch.float32, device=trainer.device).clone()
        
        feature_importance = None
        
        # Method 1: SHAP GradientExplainer
        if has_shap:
            try:
                wrapper = VFLDomainWrapper(trainer, omic_name, embeddings)
                wrapper.eval()
                
                explainer = shap.GradientExplainer(wrapper, bg_tensor)
                shap_values = explainer.shap_values(test_tensor)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                feature_importance = np.abs(np.array(shap_values)).mean(axis=0)
                logger.info(f"    {omic_name}: Using GradientExplainer")
                
            except Exception as e:
                logger.warning(f"    {omic_name}: GradientExplainer failed ({e})")
        
        # Method 2: Integrated Gradients
        if feature_importance is None:
            try:
                feature_importance = _compute_integrated_gradients(
                    silo.encoder, test_tensor, bg_tensor, trainer.device
                )
                logger.info(f"    {omic_name}: Using Integrated Gradients")
            except Exception as e:
                logger.warning(f"    {omic_name}: Integrated Gradients failed ({e})")
                feature_importance = np.zeros(silo.n_features)
        
        importance[omic_name] = feature_importance
    
    return importance


def _compute_integrated_gradients(
    encoder: nn.Module,
    features: torch.Tensor,
    baseline: torch.Tensor,
    device: torch.device,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients (Sundararajan et al., 2017).
    
    A rigorous, axiomatically-grounded attribution method.
    """
    encoder.eval()
    
    baseline_mean = baseline.mean(dim=0, keepdim=True)
    baseline_expanded = baseline_mean.expand(features.shape[0], -1)
    
    alphas = torch.linspace(0, 1, n_steps, device=device)
    integrated_grads = torch.zeros_like(features)
    
    for alpha in alphas:
        interpolated = baseline_expanded + alpha * (features - baseline_expanded)
        interpolated = interpolated.clone().detach().requires_grad_(True)
        
        output = encoder(interpolated)
        target = output.sum()
        target.backward()
        
        if interpolated.grad is not None:
            integrated_grads += interpolated.grad
    
    integrated_grads = integrated_grads / n_steps
    attributions = (features - baseline_expanded) * integrated_grads
    importance = torch.abs(attributions).mean(dim=0).detach().cpu().numpy()
    
    return importance


# =============================================================================
# Data Loading
# =============================================================================

def load_coral_data(
    data_dir: Path,
    seed: int = 42,
) -> Tuple[Dict[str, OmicSilo], np.ndarray, Dict[str, np.ndarray]]:
    """
    Load coral multi-omics data.
    
    Returns:
        silos: Dict of OmicSilo objects
        labels: Label array
        splits: Dict with train/val/test indices
    """
    processed_dir = data_dir / "processed"
    
    # Find TSV files (matching actual file names in processed/)
    omic_files = {
        "transcriptomics": "transcriptomics.tsv",
        "proteomics": "proteomics.tsv",
        "metabolomics": "metabolomics.tsv",
        "microbiome": "microbiome.tsv",
    }
    
    # Load each omic
    omics_data = {}
    sample_ids_per_omic = {}
    
    for name, filename in omic_files.items():
        filepath = processed_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, sep='\t', index_col=0)
            omics_data[name] = df.values.astype(np.float32)
            sample_ids_per_omic[name] = list(df.index)
            logger.info(f"Loaded {name}: {df.shape}")
        else:
            logger.warning(f"Missing file: {filepath}")
    
    if not omics_data:
        raise FileNotFoundError(f"No omic data found in {processed_dir}")
    
    # Find common samples
    common_samples = set(sample_ids_per_omic[list(omics_data.keys())[0]])
    for name, samples in sample_ids_per_omic.items():
        common_samples &= set(samples)
    
    common_samples = sorted(common_samples)
    logger.info(f"Common samples across all omics: {len(common_samples)}")
    
    # Align data
    silos = {}
    for name, data in omics_data.items():
        sample_ids = sample_ids_per_omic[name]
        indices = [sample_ids.index(s) for s in common_samples]
        aligned_data = data[indices]
        silos[name] = OmicSilo(name=name, features=aligned_data)
    
    # Load metadata for labels
    metadata_dir = data_dir / "metadata"
    label_file = metadata_dir / "sample_metadata.tsv"
    
    if label_file.exists():
        meta_df = pd.read_csv(label_file, sep='\t', index_col=0)
        if "treatment" in meta_df.columns:
            labels = (meta_df.loc[common_samples, "treatment"] != "Ambient").astype(int).values
        else:
            # Default: infer from sample names
            labels = np.array([1 if "HiT" in s else 0 for s in common_samples])
    else:
        # Infer from sample names: HiT = stressed (1), Amb/Field = control (0)
        # Pattern: MC-289_T1-HiT_2878 = stressed, MC-289_T1-Amb_1609 = ambient
        labels = np.array([1 if "HiT" in s else 0 for s in common_samples])
    
    logger.info(f"Labels: {sum(labels)} stressed, {len(labels) - sum(labels)} ambient")
    
    # Create splits
    np.random.seed(seed)
    n_samples = len(common_samples)
    
    if n_samples < 10:
        # Very small dataset: 60/20/20 split
        train_size = max(2, int(0.6 * n_samples))
        val_size = max(1, int(0.2 * n_samples))
    else:
        train_size = int(0.6 * n_samples)
        val_size = int(0.2 * n_samples)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    
    logger.info(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return silos, labels, splits


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NVFlare VFL Runner for Coral Multi-Omics")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/vfl", help="Output directory")
    parser.add_argument("--n-rounds", type=int, default=100, help="Training rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--compute-domain-saliency", action="store_true", help="Compute domain saliency")
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    silos, labels, splits = load_coral_data(data_dir, seed=args.seed)
    
    # Config
    config = VFLConfig(
        n_rounds=args.n_rounds,
        batch_size=args.batch_size,
        lr_client=args.lr,
        lr_server=args.lr,
        seed=args.seed,
    )
    
    # Create trainer
    trainer = NVFlareVFLTrainer(silos, labels, config)
    
    # Train
    history = trainer.train(splits["train"], splits["val"])
    
    # Evaluate on test
    test_metrics = trainer.evaluate(splits["test"], splits["train"])
    logger.info(f"Test AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save checkpoint
    checkpoint_path = output_dir / f"nvflare_vfl_seed{args.seed}.pt"
    trainer.save_checkpoint(checkpoint_path)
    
    # Domain saliency analysis
    domain_importance = {}
    if args.compute_domain_saliency:
        domain_importance = compute_domain_importance(trainer, splits["train"])
        
        # Save importance
        domain_output = output_dir / "domain_importance.json"
        domain_json = {name: imp.tolist() for name, imp in domain_importance.items()}
        with open(domain_output, 'w') as f:
            json.dump(domain_json, f, indent=2)
        logger.info(f"Saved domain importance to {domain_output}")
    
    # Save metrics
    metrics = {
        "test_auroc": test_metrics["auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "best_val_auroc": trainer.best_val_auroc,
        "n_rounds": len(history["train_loss"]),
        "config": config.__dict__,
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    main()
