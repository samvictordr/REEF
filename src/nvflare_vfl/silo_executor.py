"""
NVFlare Silo Executor for Coral Multi-omics VFL.

This executor runs on each client (silo) and manages:
- Local data loading and preprocessing
- Local model (encoder) training
- Forward pass and embedding computation  
- Gradient handling from server (split learning)
"""

from typing import Dict, Any, Optional
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from .models import OmicsEncoder, DomainFeatureSelector


logger = logging.getLogger(__name__)


class CoralSiloExecutor(Executor):
    """
    Executor for a coral omics silo in Vertical Federated Learning.
    
    Each silo handles one omics type:
    - site-transcriptomics: Gene expression data (RNA-seq)
    - site-proteomics: Protein abundance data
    - site-metabolomics: Metabolite profiles
    - site-microbiome: Microbiome composition (16S/metagenomics)
    
    The executor implements the client side of split learning:
    1. Receives sample indices from server
    2. Computes local embeddings using the encoder
    3. Sends embeddings to server
    4. Receives gradients from server
    5. Updates local encoder weights
    """
    
    def __init__(
        self,
        data_path: str = "",
        omics_type: str = "",  # Will be determined from site name if not provided
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        enable_domain_selection: bool = False,
        domain_topk: int = 500,
    ):
        super().__init__()
        
        self.data_path = data_path
        self.omics_type = omics_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.enable_domain_selection = enable_domain_selection
        self.domain_topk = domain_topk
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[OmicsEncoder] = None
        self.feature_selector: Optional[DomainFeatureSelector] = None
        self.optimizer = None
        self.local_data: Optional[np.ndarray] = None
        self.sample_ids: Optional[np.ndarray] = None
        
        # For gradient computation
        self._last_embedding: Optional[torch.Tensor] = None
        self._initialized = False
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle FL events like START_RUN, END_RUN, etc."""
        if event_type == "START_RUN":
            self._initialize(fl_ctx)
    
    def _determine_omics_type(self, fl_ctx: FLContext) -> str:
        """Determine omics type from site/client name."""
        # Try to get client name from context
        client_name = fl_ctx.get_identity_name()
        
        if not client_name:
            client_name = fl_ctx.get_prop("client_name", "")
        
        logger.info(f"Client name: {client_name}")
        
        # Map site names to omics types
        # Supports both formats: site-transcriptomics or site-A
        site_to_omics = {
            "site-transcriptomics": "transcriptomics",
            "site-proteomics": "proteomics", 
            "site-metabolomics": "metabolomics",
            "site-microbiome": "microbiome",
            "site-a": "transcriptomics",
            "site-b": "proteomics",
            "site-c": "metabolomics",
            "site-d": "microbiome",
            "transcriptomics": "transcriptomics",
            "proteomics": "proteomics",
            "metabolomics": "metabolomics",
            "microbiome": "microbiome",
        }
        
        client_lower = client_name.lower()
        for key, omics in site_to_omics.items():
            if key in client_lower:
                return omics
        
        # Fallback: use configured omics_type
        if self.omics_type:
            return self.omics_type
        
        logger.warning(f"Could not determine omics type for {client_name}, using 'unknown'")
        return "unknown"
    
    def _load_data(self, omics_type: str) -> tuple:
        """Load data for specific omics type."""
        import pandas as pd
        from pathlib import Path
        
        # Determine data directory
        if self.data_path:
            data_dir = Path(self.data_path)
        else:
            # Try to find data relative to nvflare workspace
            possible_paths = [
                Path("data/processed"),
                Path("../data/processed"),
                Path("../../data/processed"),
                Path("/home/matchaswirls/Documents/Personal Projects (Research)/Coral-holobiont-multi-omics/coral-vfl-multiomics/data/processed"),
            ]
            data_dir = None
            for p in possible_paths:
                if p.exists():
                    data_dir = p
                    break
            
            if data_dir is None:
                raise FileNotFoundError("Could not find data directory")
        
        logger.info(f"Loading {omics_type} data from {data_dir}")
        
        # File mapping
        file_map = {
            "transcriptomics": "transcriptomics_normalized.tsv",
            "proteomics": "proteomics_normalized.tsv",
            "metabolomics": "metabolomics_normalized.tsv",
            "microbiome": "microbiome_normalized.tsv",
        }
        
        filename = file_map.get(omics_type)
        if not filename:
            raise ValueError(f"Unknown omics type: {omics_type}")
        
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, sep="\t", index_col=0)
        
        # Replace NaN with 0
        df = df.fillna(0)
        
        sample_ids = df.index.values
        data = df.values.astype(np.float32)
        
        logger.info(f"Loaded {omics_type}: {data.shape[0]} samples, {data.shape[1]} features")
        
        return data, sample_ids
    
    def _initialize(self, fl_ctx: FLContext):
        """Initialize local model and data."""
        if self._initialized:
            return
        
        # Determine omics type
        self.omics_type = self._determine_omics_type(fl_ctx)
        logger.info(f"Initializing silo for omics type: {self.omics_type}")
        
        # Load data
        try:
            self.local_data, self.sample_ids = self._load_data(self.omics_type)
            input_dim = self.local_data.shape[1]
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        # Create feature selector if enabled
        if self.enable_domain_selection:
            self.feature_selector = DomainFeatureSelector(input_dim).to(self.device)
        
        # Create encoder model
        self.model = OmicsEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        self._initialized = True
        logger.info(f"Silo initialized: {self.omics_type} with {input_dim} features")
    
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Execute a task from the server."""
        
        # Ensure initialized
        self._initialize(fl_ctx)
        
        logger.debug(f"Executing task: {task_name}")
        
        if task_name == "get_embeddings":
            return self._compute_embeddings(shareable, fl_ctx)
        elif task_name == "apply_gradients":
            return self._apply_gradients(shareable, fl_ctx)
        elif task_name == "validate":
            return self._validate(shareable, fl_ctx)
        elif task_name == "update_feature_mask":
            return self._update_feature_mask(shareable, fl_ctx)
        elif task_name == "get_model_weights":
            return self._get_model_weights(fl_ctx)
        elif task_name == "set_model_weights":
            return self._set_model_weights(shareable, fl_ctx)
        else:
            logger.warning(f"Unknown task: {task_name}")
            return Shareable()
    
    def _compute_embeddings(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """
        Compute embeddings for the specified sample indices.
        
        This is the forward pass in split learning.
        """
        # Extract batch info from shareable
        dxo = shareable.get(AppConstants.DXO)
        if dxo is None:
            # Try alternate key
            dxo = shareable.get("dxo")
        
        sample_indices = None
        if dxo is not None:
            sample_indices = dxo.data.get("sample_indices")
        
        # Get data for batch
        if sample_indices is not None:
            # Map sample indices to local data
            batch_data = self.local_data[sample_indices]
        else:
            # Use all data
            batch_data = self.local_data
        
        # Convert to tensor
        batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=self.device)
        
        # Apply feature selection if enabled
        if self.feature_selector is not None:
            batch_tensor = self.feature_selector(batch_tensor)
        
        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        
        embedding = self.model(batch_tensor)
        embedding.retain_grad()  # Keep gradient for backward pass
        self._last_embedding = embedding
        
        # Return embeddings
        result_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={
                "embeddings": embedding.detach().cpu().numpy(),
                "omics_type": self.omics_type,
            },
        )
        
        result = Shareable()
        result[AppConstants.DXO] = result_dxo
        return result
    
    def _apply_gradients(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """
        Apply gradients received from server.
        
        This is the backward pass in split learning.
        """
        dxo = shareable.get(AppConstants.DXO) or shareable.get("dxo")
        
        if dxo is None or self._last_embedding is None:
            return Shareable()
        
        # Get gradients from server
        gradients = dxo.data.get("gradients")
        if gradients is None:
            return Shareable()
        
        # Convert to tensor
        grad_tensor = torch.tensor(gradients, dtype=torch.float32, device=self.device)
        
        # Backward pass
        self._last_embedding.backward(grad_tensor)
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Clear stored embedding
        self._last_embedding = None
        
        return Shareable()
    
    def _validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Compute embeddings for validation (no gradient computation)."""
        dxo = shareable.get(AppConstants.DXO) or shareable.get("dxo")
        
        sample_indices = None
        if dxo is not None:
            sample_indices = dxo.data.get("sample_indices")
        
        # Get data
        if sample_indices is not None:
            batch_data = self.local_data[sample_indices]
        else:
            batch_data = self.local_data
        
        batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=self.device)
        
        if self.feature_selector is not None:
            batch_tensor = self.feature_selector(batch_tensor)
        
        # Forward pass (no gradient)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(batch_tensor)
        
        result_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={
                "embeddings": embedding.cpu().numpy(),
                "omics_type": self.omics_type,
            },
        )
        
        result = Shareable()
        result[AppConstants.DXO] = result_dxo
        return result
    
    def _update_feature_mask(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Update feature selection mask based on gradient saliency importance."""
        dxo = shareable.get(AppConstants.DXO) or shareable.get("dxo")
        
        if dxo is None or self.feature_selector is None:
            return Shareable()
        
        importance = dxo.data.get("importance")
        topk = dxo.data.get("topk", self.domain_topk)
        
        if importance is not None:
            importance_tensor = torch.tensor(importance, device=self.device)
            self.feature_selector.update_mask(importance_tensor, topk)
            logger.info(f"Updated feature mask for {self.omics_type}: top-{topk} features")
        
        return Shareable()
    
    def _get_model_weights(self, fl_ctx: FLContext) -> Shareable:
        """Get model weights for checkpointing."""
        weights = {
            name: param.detach().cpu().numpy()
            for name, param in self.model.state_dict().items()
        }
        
        result_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={"weights": weights, "omics_type": self.omics_type},
        )
        
        result = Shareable()
        result[AppConstants.DXO] = result_dxo
        return result
    
    def _set_model_weights(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Set model weights from checkpoint."""
        dxo = shareable.get(AppConstants.DXO) or shareable.get("dxo")
        
        if dxo is not None:
            weights = dxo.data.get("weights", {})
            if weights:
                state_dict = {
                    name: torch.tensor(arr)
                    for name, arr in weights.items()
                }
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded weights for {self.omics_type}")
        
        return Shareable()
