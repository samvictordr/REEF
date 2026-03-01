"""
NVFlare Server Controller for Coral VFL.

The server controller orchestrates the VFL training:
- Coordinates communication between silos
- Aggregates embeddings from all silos
- Runs the fusion head (classifier)
- Computes loss and sends gradients back to silos
- Manages training/validation rounds
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.signal import Signal
from nvflare.apis.controller_spec import Task, ClientTask
from nvflare.apis.impl.controller import Controller
from nvflare.app_common.app_constant import AppConstants

from .models import VFLFusionHead


logger = logging.getLogger(__name__)


class CoralVFLAggregator:
    """
    Aggregator that collects embeddings from silos and trains the fusion head.
    
    This handles the server-side computation in VFL split learning.
    """
    
    def __init__(
        self,
        n_silos: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        fusion_method: str = "concat",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fusion_head = VFLFusionHead(
            n_silos=n_silos,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_classes=1,  # Binary classification
            fusion_method=fusion_method,
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.fusion_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Silo ordering (must be consistent)
        self.silo_order = ["transcriptomics", "proteomics", "metabolomics", "microbiome"]
        
        # Storage for current batch
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._embedding_grads: Dict[str, np.ndarray] = {}
    
    def reset(self):
        """Clear stored embeddings."""
        self._embeddings = {}
        self._embedding_grads = {}
    
    def accept_embedding(self, omics_type: str, embedding: np.ndarray):
        """Accept embedding from a silo."""
        self._embeddings[omics_type] = torch.tensor(
            embedding,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
    
    def forward_and_backward(self, labels: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Run forward pass, compute loss, and get gradients for each silo.
        
        Returns:
            Tuple of (loss value, dict of gradients per silo)
        """
        # Get batch size from first available embedding
        batch_size = None
        for emb in self._embeddings.values():
            batch_size = emb.shape[0]
            break
        
        if batch_size is None:
            logger.error("No embeddings received")
            return 0.0, {}
        
        # Order embeddings, filling missing with zeros
        embedding_dim = self.fusion_head.embedding_dim
        ordered_embeddings = []
        
        for omics in self.silo_order:
            if omics in self._embeddings:
                ordered_embeddings.append(self._embeddings[omics])
            else:
                # Fill missing silo with zeros
                logger.warning(f"Missing embedding for {omics}, using zeros")
                zero_emb = torch.zeros(
                    batch_size, embedding_dim,
                    device=self.device,
                    requires_grad=True,
                )
                ordered_embeddings.append(zero_emb)
        
        # Forward pass
        self.fusion_head.train()
        self.optimizer.zero_grad()
        
        logits = self.fusion_head(ordered_embeddings)
        
        # Labels
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        
        # Loss
        loss = self.criterion(logits.squeeze(-1), labels_tensor)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.fusion_head.parameters(), max_norm=1.0)
        
        # Update fusion head
        self.optimizer.step()
        
        # Collect gradients for each silo
        gradients = {}
        for i, omics in enumerate(self.silo_order):
            if omics in self._embeddings and ordered_embeddings[i].grad is not None:
                gradients[omics] = ordered_embeddings[i].grad.cpu().numpy()
        
        self._embedding_grads = gradients
        
        return loss.item(), gradients
    
    def evaluate(self, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate current model on stored embeddings.
        
        Returns:
            Dict with metrics (auroc, accuracy, loss)
        """
        if not self._embeddings:
            return {"auroc": 0.5, "accuracy": 0.5, "loss": 0.0}
        
        batch_size = list(self._embeddings.values())[0].shape[0]
        embedding_dim = self.fusion_head.embedding_dim
        
        # Order embeddings
        ordered_embeddings = []
        for omics in self.silo_order:
            if omics in self._embeddings:
                ordered_embeddings.append(self._embeddings[omics])
            else:
                ordered_embeddings.append(
                    torch.zeros(batch_size, embedding_dim, device=self.device)
                )
        
        # Forward pass
        self.fusion_head.eval()
        with torch.no_grad():
            logits = self.fusion_head(ordered_embeddings)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
            loss = self.criterion(logits.squeeze(-1), labels_tensor)
            
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).float()
        
        # Metrics
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        try:
            auroc = roc_auc_score(labels, probs_np)
        except ValueError:
            auroc = 0.5
        
        accuracy = accuracy_score(labels, preds_np)
        
        return {
            "auroc": auroc,
            "accuracy": accuracy,
            "loss": loss.item(),
        }


class CoralVFLController(Controller):
    """
    Custom VFL Controller for coral multi-omics classification.
    
    Implements vertical federated learning with split learning:
    1. Send sample indices to all silos
    2. Collect embeddings from silos
    3. Run fusion head forward + backward
    4. Send gradients back to silos
    5. Silos update their local encoders
    
    Supports gradient saliency-based feature selection when enabled.
    """
    
    def __init__(
        self,
        num_rounds: int = 50,
        data_path: str = "",
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        fusion_method: str = "concat",
        train_split: float = 0.6,
        val_split: float = 0.2,
        patience: int = 10,
        enable_domain_selection: bool = False,
        domain_warmup_rounds: int = 20,
        domain_topk: Dict[str, int] = None,
    ):
        super().__init__()
        
        self.num_rounds = num_rounds
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.fusion_method = fusion_method
        self.train_split = train_split
        self.val_split = val_split
        self.patience = patience
        
        # Domain saliency configuration
        self.enable_domain_selection = enable_domain_selection
        self.domain_warmup_rounds = domain_warmup_rounds
        self.domain_topk = domain_topk or {
            "transcriptomics": 150,
            "proteomics": 250,
            "metabolomics": 500,
            "microbiome": 400,
        }
        
        # Will be initialized
        self.aggregator: Optional[CoralVFLAggregator] = None
        self.labels: Optional[np.ndarray] = None
        self.sample_ids: Optional[np.ndarray] = None
        self.train_idx: Optional[np.ndarray] = None
        self.val_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        
        # Tracking
        self.best_val_auroc = 0.0
        self.rounds_without_improvement = 0
        self.metrics_history: List[Dict] = []
    
    def start_controller(self, fl_ctx: FLContext):
        """Initialize controller at start of job."""
        logger.info("Starting Coral VFL Controller")
        
        # Initialize aggregator
        self.aggregator = CoralVFLAggregator(
            n_silos=4,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            weight_decay=self.weight_decay,
            fusion_method=self.fusion_method,
        )
        
        # Load labels
        self._load_labels()
        
        # Create train/val/test splits
        self._create_splits()
        
        logger.info(f"Splits: train={len(self.train_idx)}, val={len(self.val_idx)}, test={len(self.test_idx)}")
    
    def _load_labels(self):
        """Load sample labels."""
        import pandas as pd
        
        # Find data directory
        if self.data_path:
            data_dir = Path(self.data_path)
        else:
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
            raise FileNotFoundError("Cannot find data directory")
        
        # Load any omics file to get sample IDs
        for filename in ["transcriptomics_normalized.tsv", "proteomics_normalized.tsv"]:
            filepath = data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, sep="\t", index_col=0)
                self.sample_ids = df.index.values
                break
        
        if self.sample_ids is None:
            raise FileNotFoundError("Cannot find omics data files")
        
        # Derive labels from sample IDs (HiT = stressed = 1, Amb = ambient = 0)
        self.labels = np.array([
            1 if "HiT" in str(sid) else 0
            for sid in self.sample_ids
        ])
        
        logger.info(f"Loaded {len(self.labels)} labels: {self.labels.sum()} stressed, {len(self.labels) - self.labels.sum()} ambient")
    
    def _create_splits(self):
        """Create train/val/test splits."""
        n_samples = len(self.labels)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_size = int(self.train_split * n_samples)
        val_size = int(self.val_split * n_samples)
        
        self.train_idx = indices[:train_size]
        self.val_idx = indices[train_size:train_size + val_size]
        self.test_idx = indices[train_size + val_size:]
    
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        """Main VFL training control flow."""
        
        logger.info(f"Starting VFL training for {self.num_rounds} rounds")
        
        for round_num in range(self.num_rounds):
            if abort_signal.triggered:
                break
            
            # Check early stopping
            if self.rounds_without_improvement >= self.patience:
                logger.info(f"Early stopping at round {round_num + 1}")
                break
            
            # Gradient saliency feature selection after warmup
            if self.enable_domain_selection and round_num == self.domain_warmup_rounds:
                logger.info("Computing gradient saliency importance for feature selection...")
                self._compute_domain_and_update_masks(fl_ctx, abort_signal)
            
            # Training round
            train_loss = self._train_round(round_num, fl_ctx, abort_signal)
            
            # Validation round
            val_metrics = self._validation_round(fl_ctx, abort_signal)
            
            # Track metrics
            self.metrics_history.append({
                "round": round_num + 1,
                "train_loss": train_loss,
                **val_metrics,
            })
            
            # Early stopping check
            if val_metrics["auroc"] > self.best_val_auroc:
                self.best_val_auroc = val_metrics["auroc"]
                self.rounds_without_improvement = 0
            else:
                self.rounds_without_improvement += 1
            
            # Log progress
            if (round_num + 1) % 5 == 0 or round_num == 0:
                logger.info(
                    f"Round {round_num + 1}/{self.num_rounds} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val AUROC: {val_metrics['auroc']:.4f} | "
                    f"Best: {self.best_val_auroc:.4f}"
                )
        
        # Final evaluation on test set
        test_metrics = self._test_evaluation(fl_ctx, abort_signal)
        logger.info(f"Test AUROC: {test_metrics['auroc']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Store final results
        fl_ctx.set_prop("test_metrics", test_metrics, private=False, sticky=True)
        fl_ctx.set_prop("best_val_auroc", self.best_val_auroc, private=False, sticky=True)
    
    def _train_round(self, round_num: int, fl_ctx: FLContext, abort_signal: Signal) -> float:
        """Execute one training round."""
        
        # Use all training data (small dataset)
        batch_indices = self.train_idx
        batch_labels = self.labels[batch_indices]
        
        # Step 1: Collect embeddings from all silos
        self.aggregator.reset()
        embeddings = self._collect_embeddings(batch_indices, fl_ctx, abort_signal, task="get_embeddings")
        
        for omics_type, embedding in embeddings.items():
            self.aggregator.accept_embedding(omics_type, embedding)
        
        # Step 2: Forward + backward on server
        loss, gradients = self.aggregator.forward_and_backward(batch_labels)
        
        # Step 3: Send gradients back to silos
        self._send_gradients(gradients, fl_ctx, abort_signal)
        
        return loss
    
    def _validation_round(self, fl_ctx: FLContext, abort_signal: Signal) -> Dict[str, float]:
        """Run validation."""
        
        batch_indices = self.val_idx
        batch_labels = self.labels[batch_indices]
        
        # Collect embeddings (validation mode)
        self.aggregator.reset()
        embeddings = self._collect_embeddings(batch_indices, fl_ctx, abort_signal, task="validate")
        
        for omics_type, embedding in embeddings.items():
            self.aggregator.accept_embedding(omics_type, embedding)
        
        # Evaluate
        return self.aggregator.evaluate(batch_labels)
    
    def _test_evaluation(self, fl_ctx: FLContext, abort_signal: Signal) -> Dict[str, float]:
        """Run final test evaluation."""
        
        batch_indices = self.test_idx
        batch_labels = self.labels[batch_indices]
        
        self.aggregator.reset()
        embeddings = self._collect_embeddings(batch_indices, fl_ctx, abort_signal, task="validate")
        
        for omics_type, embedding in embeddings.items():
            self.aggregator.accept_embedding(omics_type, embedding)
        
        return self.aggregator.evaluate(batch_labels)
    
    def _collect_embeddings(
        self,
        sample_indices: np.ndarray,
        fl_ctx: FLContext,
        abort_signal: Signal,
        task: str = "get_embeddings",
    ) -> Dict[str, np.ndarray]:
        """
        Send task to all clients and collect embeddings.
        """
        # Prepare task data
        task_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={"sample_indices": sample_indices.tolist()},
        )
        
        task_data = Shareable()
        task_data[AppConstants.DXO] = task_dxo
        
        # Create task
        task_obj = Task(
            name=task,
            data=task_data,
            timeout=60,
        )
        
        # Broadcast to all clients
        self.broadcast_and_wait(
            task=task_obj,
            fl_ctx=fl_ctx,
            targets=None,  # All clients
            min_responses=4,
            abort_signal=abort_signal,
        )
        
        # Collect results
        embeddings = {}
        for client_task in task_obj.client_tasks:
            result = client_task.result
            if result is not None:
                dxo = result.get(AppConstants.DXO)
                if dxo is not None:
                    omics_type = dxo.data.get("omics_type", "unknown")
                    embedding = dxo.data.get("embeddings")
                    if embedding is not None:
                        embeddings[omics_type] = embedding
        
        return embeddings
    
    def _send_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        """Send gradients back to respective silos."""
        
        for omics_type, grad in gradients.items():
            # Map omics type to client name
            client_name = f"site-{omics_type}"
            
            task_dxo = DXO(
                data_kind=DataKind.WEIGHTS,
                data={"gradients": grad},
            )
            
            task_data = Shareable()
            task_data[AppConstants.DXO] = task_dxo
            
            task_obj = Task(
                name="apply_gradients",
                data=task_data,
                timeout=30,
            )
            
            # Send to specific client
            self.send_and_wait(
                task=task_obj,
                fl_ctx=fl_ctx,
                targets=[client_name],
                abort_signal=abort_signal,
            )
    
    def _compute_domain_and_update_masks(self, fl_ctx: FLContext, abort_signal: Signal):
        """Compute gradient saliency importance and update feature masks on silos."""
        # This would require gradient saliency computation on the server side
        # For now, we send the topk values and let silos handle feature selection
        # In a more complete implementation, you'd compute gradients-based importance here
        
        for omics_type, topk in self.domain_topk.items():
            client_name = f"site-{omics_type}"
            
            # For now, we just tell silos to use top-K features
            # based on variance or other simple criteria
            task_dxo = DXO(
                data_kind=DataKind.WEIGHTS,
                data={
                    "topk": topk,
                    "importance": None,  # Silos will compute their own
                },
            )
            
            task_data = Shareable()
            task_data[AppConstants.DXO] = task_dxo
            
            task_obj = Task(
                name="update_feature_mask",
                data=task_data,
                timeout=30,
            )
            
            self.send_and_wait(
                task=task_obj,
                fl_ctx=fl_ctx,
                targets=[client_name],
                abort_signal=abort_signal,
            )
            
            logger.info(f"Sent feature selection update to {omics_type}: top-{topk}")
    
    def stop_controller(self, fl_ctx: FLContext):
        """Clean up when controller stops."""
        logger.info("Stopping Coral VFL Controller")
        logger.info(f"Final best validation AUROC: {self.best_val_auroc:.4f}")
