"""
Vertical Federated Learning models for coral multi-omics.

Implements the VFL paradigm where:
- Each silo (lab) holds different omics blocks
- Silos encode their local data and send embeddings (not raw features) to server
- Server fuses embeddings and computes gradients
- Gradients flow back to silos for local updates

This module provides:
- OmicsEncoder: Per-silo encoder
- VFLFusionHead: Server-side fusion and prediction
- VFLModel: Complete VFL model for local simulation
- MosaicVFLModel: VFL with mosaic sample support
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OmicsEncoder(nn.Module):
    """
    Encoder for a single omics block (runs on a silo/client).
    
    Transforms raw features into a fixed-size embedding that can be
    shared with the server without revealing raw data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize encoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            n_layers: Number of hidden layers
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final projection to embedding
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features to embedding.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            Embedding (batch_size, embedding_dim)
        """
        return self.encoder(x)


class VFLFusionHead(nn.Module):
    """
    Fusion head for VFL (runs on the server/aggregator).
    
    Receives embeddings from all silos and produces predictions.
    Only the server has access to labels for loss computation.
    """
    
    def __init__(
        self,
        n_silos: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 2,
        task: str = "classification",
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        fusion_method: str = "concat",  # 'concat', 'sum', 'attention', 'gated'
    ):
        """
        Initialize fusion head.
        
        Args:
            n_silos: Number of silos (omics blocks)
            embedding_dim: Dimension of each silo's embedding
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes (1 for regression)
            task: 'classification' or 'regression'
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
            fusion_method: How to combine embeddings
        """
        super().__init__()
        
        self.n_silos = n_silos
        self.embedding_dim = embedding_dim
        self.task = task
        self.n_classes = n_classes if task == "classification" else 1
        self.fusion_method = fusion_method
        
        # Determine fusion input dimension
        if fusion_method == "concat":
            fusion_input_dim = embedding_dim * n_silos
        elif fusion_method in ["sum", "attention", "gated"]:
            fusion_input_dim = embedding_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Optional attention mechanism
        if fusion_method == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Optional gating mechanism
        if fusion_method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * n_silos, n_silos),
                nn.Softmax(dim=-1)
            )
        
        # Fusion network
        layers = [
            nn.Linear(fusion_input_dim, hidden_dim),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        ])
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        
        self.fusion_net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim // 2, self.n_classes)
    
    def fuse(
        self,
        embeddings: Union[List[torch.Tensor], torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse embeddings from all silos.
        
        Args:
            embeddings: List of embeddings or stacked tensor (batch, n_silos, emb_dim)
            mask: Optional mask (batch, n_silos) indicating available silos
        
        Returns:
            Fused representation (batch, fusion_dim)
        """
        if isinstance(embeddings, list):
            stacked = torch.stack(embeddings, dim=1)
        else:
            stacked = embeddings
        
        batch_size = stacked.shape[0]
        
        if self.fusion_method == "concat":
            fused = stacked.view(batch_size, -1)
            if mask is not None:
                # Zero out missing embeddings
                mask_expanded = mask.unsqueeze(-1).expand_as(stacked)
                stacked_masked = stacked * mask_expanded
                fused = stacked_masked.view(batch_size, -1)
        
        elif self.fusion_method == "sum":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                fused = (stacked * mask_expanded).sum(dim=1)
                n_available = mask.sum(dim=1, keepdim=True).clamp(min=1)
                fused = fused / n_available
            else:
                fused = stacked.mean(dim=1)
        
        elif self.fusion_method == "attention":
            if mask is not None:
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            
            attn_out, _ = self.attention(
                stacked, stacked, stacked,
                key_padding_mask=attn_mask
            )
            fused = attn_out.mean(dim=1)
        
        elif self.fusion_method == "gated":
            # Compute gates based on all embeddings
            flat = stacked.view(batch_size, -1)
            gates = self.gate(flat)  # (batch, n_silos)
            
            if mask is not None:
                # Zero out gates for missing silos and renormalize
                gates = gates * mask
                gates = gates / gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
            
            # Weighted combination
            fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)
        
        return fused
    
    def forward(
        self,
        embeddings: Union[List[torch.Tensor], torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            embeddings: List of silo embeddings or stacked tensor
            mask: Optional availability mask
            return_embedding: Whether to return fusion embedding
        
        Returns:
            Logits (and optionally embedding)
        """
        fused = self.fuse(embeddings, mask)
        embedding = self.fusion_net(fused)
        logits = self.head(embedding)
        
        if return_embedding:
            return logits, embedding
        return logits
    
    def predict(
        self,
        embeddings: Union[List[torch.Tensor], torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get predictions."""
        logits = self.forward(embeddings, mask)
        
        if self.task == "classification":
            if self.n_classes == 2:
                return torch.sigmoid(logits.squeeze(-1))
            return F.softmax(logits, dim=-1)
        return logits.squeeze(-1)
    
    def get_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss."""
        if self.task == "classification":
            if self.n_classes == 2:
                return F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),
                    targets.float(),
                    weight=class_weights
                )
            return F.cross_entropy(logits, targets.long(), weight=class_weights)
        return F.mse_loss(logits.squeeze(-1), targets.float())


class VFLModel(nn.Module):
    """
    Complete VFL model for local simulation.
    
    Combines per-silo encoders with the server fusion head.
    Useful for:
    - Local testing and debugging
    - Baseline comparisons
    - Understanding the VFL architecture
    
    In actual federated deployment, encoders run on clients
    and fusion head runs on server.
    """
    
    def __init__(
        self,
        omics_dims: Dict[str, int],
        encoder_hidden_dim: int = 128,
        embedding_dim: int = 64,
        fusion_hidden_dim: int = 128,
        n_classes: int = 2,
        task: str = "classification",
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        fusion_method: str = "concat",
    ):
        """
        Initialize VFL model.
        
        Args:
            omics_dims: Dict mapping omics name to feature dimension
            encoder_hidden_dim: Hidden dim for encoders
            embedding_dim: Embedding dimension
            fusion_hidden_dim: Hidden dim for fusion head
            n_classes: Number of output classes
            task: 'classification' or 'regression'
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
            fusion_method: Fusion strategy
        """
        super().__init__()
        
        self.omics_names = list(omics_dims.keys())
        self.omics_dims = omics_dims
        self.embedding_dim = embedding_dim
        self.task = task
        self.n_classes = n_classes
        
        # Create per-omics encoders (would run on clients)
        self.encoders = nn.ModuleDict()
        for name, dim in omics_dims.items():
            self.encoders[name] = OmicsEncoder(
                input_dim=dim,
                hidden_dim=encoder_hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
        
        # Create fusion head (would run on server)
        self.fusion_head = VFLFusionHead(
            n_silos=len(omics_dims),
            embedding_dim=embedding_dim,
            hidden_dim=fusion_hidden_dim,
            n_classes=n_classes,
            task=task,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            fusion_method=fusion_method,
        )
    
    def encode(
        self,
        x: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Encode all omics blocks.
        
        Args:
            x: Dict of omics tensors
        
        Returns:
            Dict of embeddings and stacked tensor
        """
        embeddings = {}
        for name in self.omics_names:
            if name in x:
                embeddings[name] = self.encoders[name](x[name])
            else:
                batch_size = list(x.values())[0].shape[0]
                device = list(x.values())[0].device
                embeddings[name] = torch.zeros(
                    batch_size, self.embedding_dim, device=device
                )
        
        stacked = torch.stack(
            [embeddings[name] for name in self.omics_names],
            dim=1
        )
        
        return embeddings, stacked
    
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through full VFL pipeline.
        
        Args:
            x: Dict of omics tensors
            mask: Optional availability mask
            return_embedding: Whether to return fusion embedding
        """
        _, stacked = self.encode(x)
        return self.fusion_head(stacked, mask, return_embedding)
    
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get predictions."""
        _, stacked = self.encode(x)
        return self.fusion_head.predict(stacked, mask)
    
    def get_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss."""
        return self.fusion_head.get_loss(logits, targets, class_weights)
    
    def get_encoder(self, omic_name: str) -> OmicsEncoder:
        """Get encoder for a specific omics (for deployment)."""
        return self.encoders[omic_name]
    
    def get_fusion_head(self) -> VFLFusionHead:
        """Get fusion head (for server deployment)."""
        return self.fusion_head


class MosaicVFLModel(VFLModel):
    """
    VFL model with support for mosaic (incomplete) samples.
    
    Extends VFLModel with:
    - Learnable imputation for missing omics
    - Robust fusion that handles varying availability patterns
    """
    
    def __init__(
        self,
        omics_dims: Dict[str, int],
        imputation: str = "learned",  # 'zero', 'learned'
        **kwargs
    ):
        """
        Initialize mosaic VFL model.
        
        Args:
            omics_dims: Dict mapping omics name to feature dimension
            imputation: Strategy for missing omics
            **kwargs: Additional args for VFLModel
        """
        super().__init__(omics_dims, **kwargs)
        
        self.imputation = imputation
        
        if imputation == "learned":
            # Learnable default embeddings for missing omics
            self.default_embeddings = nn.ParameterDict({
                name: nn.Parameter(torch.randn(1, self.embedding_dim) * 0.1)
                for name in self.omics_names
            })
    
    def encode(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Encode with imputation for missing omics.
        
        Args:
            x: Dict of omics tensors (may have missing keys)
            mask: Availability mask (batch, n_omics)
        
        Returns:
            Dict of embeddings and stacked tensor
        """
        embeddings = {}
        batch_size = list(x.values())[0].shape[0]
        device = list(x.values())[0].device
        
        for i, name in enumerate(self.omics_names):
            if name in x:
                emb = self.encoders[name](x[name])
            else:
                # Use imputation
                if self.imputation == "learned":
                    emb = self.default_embeddings[name].expand(batch_size, -1)
                else:
                    emb = torch.zeros(batch_size, self.embedding_dim, device=device)
            
            # Apply per-sample availability mask
            if mask is not None:
                available = mask[:, i:i+1]  # (batch, 1)
                
                if self.imputation == "learned":
                    # Blend: use embedding where available, default where not
                    default = self.default_embeddings[name].expand(batch_size, -1).to(device)
                    emb = available * emb + (1 - available) * default
                else:
                    emb = emb * available
            
            embeddings[name] = emb
        
        stacked = torch.stack(
            [embeddings[name] for name in self.omics_names],
            dim=1
        )
        
        return embeddings, stacked
    
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with mosaic support."""
        _, stacked = self.encode(x, mask)
        return self.fusion_head(stacked, mask, return_embedding)


class GroupedVFLModel(nn.Module):
    """
    VFL model with biologically grouped omics.
    
    Groups omics into biologically meaningful clusters:
    - Genetic: transcriptomics (DNA → RNA expression)
    - Functional: proteomics, metabolomics (downstream of transcription)
    - Microbiome: symbiont and bacterial communities
    - Environment: abiotic factors
    
    This architecture respects approximate causal structure and
    reduces spurious correlations between unrelated modalities.
    """
    
    # Default biological groupings
    DEFAULT_GROUPS = {
        "genetic": ["transcriptomics"],
        "functional": ["proteomics", "metabolomics"],
        "microbiome": ["microbiome"],
        "environment": ["environment"],
    }
    
    def __init__(
        self,
        omics_dims: Dict[str, int],
        groups: Optional[Dict[str, List[str]]] = None,
        encoder_hidden_dim: int = 128,
        embedding_dim: int = 64,
        group_embedding_dim: int = 64,
        fusion_hidden_dim: int = 128,
        n_classes: int = 2,
        task: str = "classification",
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        """
        Initialize grouped VFL model.
        
        Args:
            omics_dims: Dict mapping omics name to feature dimension
            groups: Dict mapping group name to list of omics names
            encoder_hidden_dim: Hidden dim for per-omics encoders
            embedding_dim: Per-omics embedding dimension
            group_embedding_dim: Per-group embedding dimension
            fusion_hidden_dim: Hidden dim for final fusion
            n_classes: Number of output classes
            task: 'classification' or 'regression'
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        self.omics_names = list(omics_dims.keys())
        self.omics_dims = omics_dims
        self.embedding_dim = embedding_dim
        self.group_embedding_dim = group_embedding_dim
        self.task = task
        self.n_classes = n_classes
        
        # Determine groups (filter to available omics)
        if groups is None:
            groups = {
                k: [o for o in v if o in omics_dims]
                for k, v in self.DEFAULT_GROUPS.items()
            }
        groups = {k: v for k, v in groups.items() if v}
        self.groups = groups
        self.group_names = list(groups.keys())
        
        # Per-omics encoders
        self.encoders = nn.ModuleDict()
        for name, dim in omics_dims.items():
            self.encoders[name] = OmicsEncoder(
                input_dim=dim,
                hidden_dim=encoder_hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
        
        # Per-group fusion (fuses omics within each biological group)
        self.group_fusers = nn.ModuleDict()
        for group_name, group_omics in groups.items():
            n_omics_in_group = len(group_omics)
            if n_omics_in_group > 0:
                self.group_fusers[group_name] = nn.Sequential(
                    nn.Linear(embedding_dim * n_omics_in_group, group_embedding_dim),
                    nn.BatchNorm1d(group_embedding_dim) if use_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
        
        # Final fusion across groups
        n_groups = len(groups)
        self.final_fusion = nn.Sequential(
            nn.Linear(group_embedding_dim * n_groups, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.head = nn.Linear(
            fusion_hidden_dim // 2,
            n_classes if task == "classification" else 1
        )
    
    def encode_groups(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode and fuse within biological groups.
        
        Args:
            x: Dict of omics tensors
            mask: Optional availability mask (batch, n_omics)
        
        Returns:
            Dict of group embeddings
        """
        batch_size = list(x.values())[0].shape[0]
        device = list(x.values())[0].device
        
        # First encode all omics
        omics_embeddings = {}
        for i, name in enumerate(self.omics_names):
            if name in x:
                omics_embeddings[name] = self.encoders[name](x[name])
            else:
                omics_embeddings[name] = torch.zeros(
                    batch_size, self.embedding_dim, device=device
                )
        
        # Then fuse within groups
        group_embeddings = {}
        for group_name, group_omics in self.groups.items():
            # Concatenate omics within this group
            group_embeds = [omics_embeddings[o] for o in group_omics]
            group_cat = torch.cat(group_embeds, dim=-1)
            
            # Apply group fusion
            group_embeddings[group_name] = self.group_fusers[group_name](group_cat)
        
        return group_embeddings
    
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with grouped fusion."""
        group_embeddings = self.encode_groups(x, mask)
        
        # Concatenate group embeddings
        fused = torch.cat(
            [group_embeddings[g] for g in self.group_names],
            dim=-1
        )
        
        embedding = self.final_fusion(fused)
        logits = self.head(embedding)
        
        if return_embedding:
            return logits, embedding
        return logits
    
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get predictions."""
        logits = self.forward(x, mask)
        
        if self.task == "classification":
            if self.n_classes == 2:
                return torch.sigmoid(logits.squeeze(-1))
            return F.softmax(logits, dim=-1)
        return logits.squeeze(-1)
    
    def get_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss."""
        if self.task == "classification":
            if self.n_classes == 2:
                return F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),
                    targets.float(),
                    weight=class_weights
                )
            return F.cross_entropy(logits, targets.long(), weight=class_weights)
        return F.mse_loss(logits.squeeze(-1), targets.float())


def create_vfl_model(
    omics_dims: Dict[str, int],
    architecture: str = "vfl",
    task: str = "classification",
    **kwargs
) -> nn.Module:
    """
    Factory function to create VFL models.
    
    Args:
        omics_dims: Dict mapping omics name to feature dimension
        architecture: 'vfl', 'mosaic_vfl', 'grouped_vfl'
        task: 'classification' or 'regression'
        **kwargs: Additional arguments passed to model
    
    Returns:
        Model instance
    """
    if architecture == "vfl":
        return VFLModel(omics_dims, task=task, **kwargs)
    elif architecture == "mosaic_vfl":
        return MosaicVFLModel(omics_dims, task=task, **kwargs)
    elif architecture == "grouped_vfl":
        return GroupedVFLModel(omics_dims, task=task, **kwargs)
    else:
        raise ValueError(f"Unknown VFL architecture: {architecture}")
