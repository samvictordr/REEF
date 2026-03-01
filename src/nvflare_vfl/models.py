"""
Neural network models for NVFlare VFL.

Models for each silo (encoder) and the server (fusion head).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class OmicsEncoder(nn.Module):
    """
    Local encoder for a single omics type.
    
    Runs on each client silo and produces embeddings
    that are sent to the server for fusion.
    
    Uses LayerNorm instead of BatchNorm for stability with small batches.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Feature compression (for high-dimensional omics)
        self.compress = nn.Linear(input_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing embedding."""
        h = self.compress(x)
        return self.net(h)


class VFLFusionHead(nn.Module):
    """
    Fusion head that combines embeddings from all silos.
    
    Runs on the server and produces the final prediction.
    Supports multiple fusion strategies: concat, attention, or gated.
    """
    
    def __init__(
        self,
        n_silos: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 1,  # Binary classification
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
                dropout=0.1,
            )
        elif fusion_method == "gated":
            input_dim = embedding_dim
            self.gates = nn.Sequential(
                nn.Linear(n_silos * embedding_dim, n_silos),
                nn.Softmax(dim=-1),
            )
        else:
            input_dim = embedding_dim
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        embeddings: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: List of embeddings from each silo [B, embedding_dim] each
            mask: Optional availability mask for missing data [B, n_silos]
        
        Returns:
            Logits [B, n_classes]
        """
        # Stack embeddings: [B, n_silos, embedding_dim]
        stacked = torch.stack(embeddings, dim=1)
        
        if mask is not None:
            # Zero out unavailable silos
            mask = mask.unsqueeze(-1)  # [B, n_silos, 1]
            stacked = stacked * mask
        
        if self.fusion_method == "concat":
            # Concatenate all embeddings
            fused = stacked.view(stacked.shape[0], -1)  # [B, n_silos * embedding_dim]
            
        elif self.fusion_method == "attention":
            # Self-attention across silos
            fused, _ = self.attention(stacked, stacked, stacked)
            fused = fused.mean(dim=1)  # [B, embedding_dim]
            
        elif self.fusion_method == "gated":
            # Gated fusion
            flat = stacked.view(stacked.shape[0], -1)
            gates = self.gates(flat)  # [B, n_silos]
            gates = gates.unsqueeze(-1)  # [B, n_silos, 1]
            fused = (stacked * gates).sum(dim=1)  # [B, embedding_dim]
            
        else:
            # Simple averaging
            fused = stacked.mean(dim=1)
        
        return self.classifier(fused)


class DomainFeatureSelector(nn.Module):
    """
    Feature selector based on domain saliency.
    
    Creates a learnable mask that can be applied to input features
    based on domain-computed importance scores.
    """
    
    def __init__(self, input_dim: int, initial_mask: Optional[torch.Tensor] = None):
        super().__init__()
        
        if initial_mask is not None:
            self.mask = nn.Parameter(initial_mask, requires_grad=False)
        else:
            self.mask = nn.Parameter(torch.ones(input_dim), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature mask."""
        return x * self.mask
    
    def update_mask(self, importance: torch.Tensor, topk: int):
        """Update mask based on domain saliency scores."""
        _, top_indices = torch.topk(importance, min(topk, len(importance)))
        new_mask = torch.zeros_like(self.mask)
        new_mask[top_indices] = 1.0
        self.mask.data = new_mask
