"""
NVFlare Server Controller for Coral VFL.

The server:
- Holds the fusion head (combines embeddings from all silos)
- Has access to labels
- Orchestrates training rounds
- Computes loss and sends gradients back to silos
"""

from typing import Dict, List, Any, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.workflows.splitnn_workflow import SplitNNWorkflow
from nvflare.app_common.abstract.aggregator import Aggregator


logger = logging.getLogger(__name__)


class VFLFusionHead(nn.Module):
    """Fusion head that combines embeddings from all silos."""
    
    def __init__(
        self,
        n_silos: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 2,
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
        else:
            input_dim = embedding_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes if n_classes > 2 else 1),
        )
    
    def forward(
        self, 
        embeddings: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: List of embeddings from each silo
            mask: Availability mask for mosaic data
        
        Returns:
            Logits
        """
        stacked = torch.stack(embeddings, dim=1)  # (B, n_silos, embedding_dim)
        
        if mask is not None:
            # Zero out unavailable silos
            mask = mask.unsqueeze(-1)  # (B, n_silos, 1)
            stacked = stacked * mask
        
        if self.fusion_method == "concat":
            fused = stacked.view(stacked.shape[0], -1)
        elif self.fusion_method == "attention":
            fused, _ = self.attention(stacked, stacked, stacked)
            fused = fused.mean(dim=1)
        else:
            fused = stacked.mean(dim=1)
        
        return self.classifier(fused)


class CoralVFLAggregator(Aggregator):
    """
    Aggregator for coral VFL.
    
    Instead of averaging weights (as in HFL), this aggregator:
    1. Collects embeddings from all silos
    2. Runs the fusion head
    3. Computes loss
    4. Sends gradients back
    """
    
    def __init__(
        self,
        n_silos: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 2,
        lr: float = 0.001,
    ):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fusion_head = VFLFusionHead(
            n_silos=n_silos,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.fusion_head.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Storage for embeddings and results
        self._embeddings = {}
        self._labels = None
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle FL events."""
        pass
    
    def accept(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
    ) -> bool:
        """Accept a contribution from a client."""
        client_name = fl_ctx.get_prop("client_name", "unknown")
        
        dxo = shareable.get("dxo")
        if dxo and "embeddings" in dxo.data:
            embedding = torch.tensor(
                dxo.data["embeddings"], 
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self._embeddings[client_name] = embedding
            return True
        
        return False
    
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """
        Aggregate embeddings and compute gradients.
        
        Returns:
            Shareable containing gradients for each silo
        """
        if len(self._embeddings) < 4:
            logger.warning(f"Not all silos contributed: {len(self._embeddings)}/4")
        
        # Order embeddings by silo name
        ordered_embeddings = [
            self._embeddings.get(f"site-{chr(ord('A') + i)}")
            for i in range(4)
        ]
        
        # Replace None with zeros
        batch_size = None
        for emb in ordered_embeddings:
            if emb is not None:
                batch_size = emb.shape[0]
                break
        
        if batch_size is None:
            return Shareable()
        
        for i, emb in enumerate(ordered_embeddings):
            if emb is None:
                ordered_embeddings[i] = torch.zeros(
                    batch_size, self.fusion_head.embedding_dim,
                    device=self.device, requires_grad=True,
                )
        
        # Forward pass
        self.fusion_head.train()
        self.optimizer.zero_grad()
        
        logits = self.fusion_head(ordered_embeddings)
        
        # Get labels
        if self._labels is None:
            logger.warning("No labels available, using placeholder")
            self._labels = torch.zeros(batch_size, device=self.device)
        
        labels = self._labels[:batch_size].float()
        loss = self.criterion(logits.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Collect gradients for each silo
        gradients = {}
        for i, emb in enumerate(ordered_embeddings):
            if emb.grad is not None:
                site_name = f"site-{chr(ord('A') + i)}"
                gradients[site_name] = emb.grad.cpu().numpy()
        
        # Create result shareable
        result_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={
                "gradients": gradients,
                "loss": loss.item(),
            },
        )
        
        result = Shareable()
        result["dxo"] = result_dxo
        
        # Clear storage for next round
        self._embeddings = {}
        
        return result
    
    def set_labels(self, labels: np.ndarray):
        """Set labels for current batch."""
        self._labels = torch.tensor(labels, dtype=torch.float32, device=self.device)


class CoralVFLController(SplitNNWorkflow):
    """
    Custom VFL controller for coral multi-omics.
    
    Extends SplitNNWorkflow with coral-specific logic.
    """
    
    def __init__(
        self,
        min_clients: int = 4,
        num_rounds: int = 50,
        **kwargs,
    ):
        super().__init__(
            min_clients=min_clients,
            num_rounds=num_rounds,
            **kwargs,
        )
        
        self.aggregator = CoralVFLAggregator()
        self._data_loader = None
    
    def control_flow(self, abort_signal, fl_ctx: FLContext):
        """Main control flow for VFL training."""
        engine = fl_ctx.get_engine()
        
        # Get data loader if available
        self._data_loader = engine.get_component("data_loader")
        
        for round_num in range(self.num_rounds):
            if abort_signal.triggered:
                break
            
            logger.info(f"Starting round {round_num + 1}/{self.num_rounds}")
            
            # Get batch indices
            if self._data_loader:
                train_indices = self._data_loader.get_sample_indices("train")
                batch_size = min(32, len(train_indices))
                batch_indices = np.random.choice(
                    train_indices, size=batch_size, replace=False
                )
                labels = self._data_loader.load_labels()[batch_indices]
                self.aggregator.set_labels(labels)
            else:
                batch_indices = np.arange(32)
                self.aggregator.set_labels(np.random.randint(0, 2, 32))
            
            # Request embeddings from all silos
            task_data = DXO(
                data_kind=DataKind.WEIGHTS,
                data={"batch_indices": batch_indices.tolist()},
            )
            
            shareable = Shareable()
            shareable["dxo"] = task_data
            
            # Send task to clients and collect responses
            responses = self.broadcast_and_wait(
                task_name="train",
                task_data=shareable,
                min_responses=self.min_clients,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
            
            # Aggregate embeddings
            for client_name, response in responses.items():
                self.aggregator.accept(response, fl_ctx)
            
            agg_result = self.aggregator.aggregate(fl_ctx)
            
            # Send gradients back to clients
            if "dxo" in agg_result:
                gradients_data = agg_result["dxo"].data.get("gradients", {})
                loss = agg_result["dxo"].data.get("loss", 0)
                
                logger.info(f"Round {round_num + 1} loss: {loss:.4f}")
                
                for client_name, grad in gradients_data.items():
                    grad_dxo = DXO(
                        data_kind=DataKind.WEIGHTS,
                        data={"gradients": grad},
                    )
                    grad_shareable = Shareable()
                    grad_shareable["dxo"] = grad_dxo
                    
                    self.send_to_client(
                        task_name="train",
                        task_data=grad_shareable,
                        client_name=client_name,
                        fl_ctx=fl_ctx,
                    )
        
        logger.info("VFL training completed")
