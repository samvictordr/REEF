"""
NVFlare Silo Executor for Coral Multi-omics VFL.

This executor runs on each client (silo) and manages:
- Local data loading
- Local model (encoder)
- Forward pass and embedding computation
- Gradient handling from server
"""

from typing import Dict, Any

import torch
import torch.nn as nn
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.signal import Signal


class OmicsEncoder(nn.Module):
    """Local encoder for a single omics type."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoralSiloExecutor(Executor):
    """
    Executor for a coral omics silo in VFL.
    
    Each silo handles one omics type:
    - site-A: transcriptomics
    - site-B: proteomics
    - site-C: metabolomics
    - site-D: microbiome
    """
    
    def __init__(
        self,
        lr: float = 0.001,
        epochs: int = 1,
        batch_size: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.data_loader = None
        self._initialized = False
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle FL events."""
        pass
    
    def _initialize(self, fl_ctx: FLContext):
        """Initialize local model and data."""
        if self._initialized:
            return
        
        # Get site name to determine omics type
        site_name = fl_ctx.get_prop("site_name", "unknown")
        
        # Map site to omics
        site_to_omics = {
            "site-A": "transcriptomics",
            "site-B": "proteomics",
            "site-C": "metabolomics",
            "site-D": "microbiome",
        }
        
        self.omics_type = site_to_omics.get(site_name, "unknown")
        
        # Get data loader component
        engine = fl_ctx.get_engine()
        data_loader = engine.get_component("data_loader")
        
        if data_loader:
            self.local_data = data_loader.load_omics(self.omics_type)
            input_dim = self.local_data.shape[1]
        else:
            # Fallback for testing
            input_dim = 1000
        
        # Create model
        self.model = OmicsEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self._initialized = True
    
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Execute a task."""
        
        self._initialize(fl_ctx)
        
        if task_name == "train":
            return self._train(shareable, fl_ctx)
        elif task_name == "validate":
            return self._validate(shareable, fl_ctx)
        elif task_name == "submit_model":
            return self._submit_model(shareable, fl_ctx)
        else:
            return Shareable()
    
    def _train(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """
        Training task: compute embeddings and handle gradients.
        
        In VFL split learning:
        1. Receive sample indices or data from server
        2. Compute local embeddings
        3. Send embeddings to server
        4. Receive gradients from server
        5. Update local model
        """
        # Extract batch indices from shareable
        dxo = shareable.get("dxo")
        if dxo:
            batch_indices = dxo.data.get("batch_indices", None)
            gradients = dxo.data.get("gradients", None)
        else:
            batch_indices = None
            gradients = None
        
        self.model.train()
        
        if gradients is not None:
            # Backward pass: apply gradients from server
            grad_tensor = torch.tensor(gradients, device=self.device)
            
            # The stored embeddings should have retain_grad enabled
            if hasattr(self, "_last_embedding") and self._last_embedding is not None:
                self._last_embedding.backward(grad_tensor)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            return Shareable()
        
        else:
            # Forward pass: compute embeddings
            if batch_indices is not None:
                batch_data = torch.tensor(
                    self.local_data[batch_indices], 
                    dtype=torch.float32, 
                    device=self.device
                )
            else:
                # Use all data if no indices specified
                batch_data = torch.tensor(
                    self.local_data, 
                    dtype=torch.float32, 
                    device=self.device
                )
            
            self.optimizer.zero_grad()
            embedding = self.model(batch_data)
            embedding.retain_grad()
            self._last_embedding = embedding
            
            # Send embeddings to server
            result_dxo = DXO(
                data_kind=DataKind.WEIGHTS,
                data={"embeddings": embedding.detach().cpu().numpy()},
            )
            
            result = Shareable()
            result["dxo"] = result_dxo
            return result
    
    def _validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Validation task: compute embeddings for validation set."""
        self.model.eval()
        
        with torch.no_grad():
            val_data = torch.tensor(
                self.local_data, 
                dtype=torch.float32, 
                device=self.device
            )
            embedding = self.model(val_data)
        
        result_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={"embeddings": embedding.cpu().numpy()},
        )
        
        result = Shareable()
        result["dxo"] = result_dxo
        return result
    
    def _submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Submit model weights."""
        weights = {
            name: param.detach().cpu().numpy() 
            for name, param in self.model.state_dict().items()
        }
        
        result_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=weights,
        )
        
        result = Shareable()
        result["dxo"] = result_dxo
        return result
