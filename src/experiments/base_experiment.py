"""
Base experiment class for coral multi-omics VFL experiments.

Provides common functionality for:
- Configuration loading
- Data loading and splitting
- Training loops
- Evaluation and logging
"""

import logging
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import (
    CoralMultiOmicsLoader,
    CoralMultiOmicsDataset,
    MosaicMultiOmicsDataset,
    MultiOmicsPreprocessor,
    create_cv_splits,
    DataSplitter,
)
from ..evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    MetricsTracker,
)


logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """
    Base class for all experiments.
    
    Handles common setup, training, and evaluation logic.
    Subclasses implement specific model creation and experiment design.
    """
    
    def __init__(
        self,
        config: Union[str, Path, Dict],
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize experiment.
        
        Args:
            config: Config file path or dict
            output_dir: Output directory for results
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Load configuration
        if isinstance(config, (str, Path)):
            self.config = self._load_config(config)
            self.config_path = Path(config)
        else:
            self.config = config
            self.config_path = None
        
        # Set up output directory
        self.output_dir = Path(output_dir or self.config.get("output_dir", "outputs"))
        self.experiment_name = self.config.get("experiment_name", self.__class__.__name__)
        self.run_dir = self.output_dir / self.experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Output dir: {self.run_dir}")
        logger.info(f"Device: {self.device}")
        
        # Save config
        self._save_config()
        
        # Data and model placeholders
        self.data = None
        self.preprocessor = None
        self.splitter = None
    
    def _load_config(self, path: Union[str, Path]) -> Dict:
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        return config
    
    def _save_config(self):
        """Save configuration to run directory."""
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _setup_logging(self):
        """Set up logging to file and console."""
        log_path = self.run_dir / "experiment.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    def load_data(self):
        """Load and preprocess data."""
        data_config = self.config.get("data", {})
        data_root = Path(data_config.get("data_root", "data"))
        
        logger.info(f"Loading data from {data_root}")
        
        # Load raw data
        loader = CoralMultiOmicsLoader(
            data_root,
            omics_to_load=data_config.get("omics", None),
        )
        self.data = loader.load(align_samples=data_config.get("align_samples", True))
        
        logger.info(f"Loaded {self.data.n_samples} samples, {self.data.n_omics} omics")
        
        # Preprocess
        if data_config.get("preprocess", True):
            self.preprocessor = MultiOmicsPreprocessor(
                omics_configs=data_config.get("preprocessing", {})
            )
            self.data = self.preprocessor.fit_transform(self.data)
            logger.info("Preprocessing complete")
        
        # Set up splitter
        self.splitter = DataSplitter(
            self.data,
            n_folds=data_config.get("n_folds", 5),
            random_state=data_config.get("random_state", 42),
        )
        
        return self.data
    
    def get_omics_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each omics block."""
        return {
            name: block.n_features
            for name, block in self.data.blocks.items()
        }
    
    @abstractmethod
    def create_model(self, **kwargs) -> nn.Module:
        """Create the model for this experiment. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def run(self) -> Dict:
        """Run the experiment. Must be implemented by subclass."""
        pass
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function (optional, uses model.get_loss if None)
        
        Returns:
            Tuple of (average loss, average metric)
        """
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Handle different batch formats
            if len(batch) == 2:
                X, y = batch
                mask = None
            elif len(batch) == 3:
                X, mask, y = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            # Move to device
            if isinstance(X, dict):
                X = {k: v.to(self.device) for k, v in X.items()}
            else:
                X = X.to(self.device)
            
            if mask is not None:
                mask = mask.to(self.device)
            
            y = y.to(self.device)
            
            # Forward pass
            if mask is not None:
                logits = model(X, mask)
            else:
                logits = model(X)
            
            # Compute loss
            if criterion is not None:
                loss = criterion(logits.squeeze(), y)
            else:
                loss = model.get_loss(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions
            with torch.no_grad():
                if mask is not None:
                    preds = model.predict(X, mask)
                else:
                    preds = model.predict(X)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Compute metric
        task = self.config.get("task", "classification")
        if task == "classification":
            metrics = compute_classification_metrics(
                np.array(all_targets), np.array(all_preds)
            )
            avg_metric = metrics.auroc
        else:
            metrics = compute_regression_metrics(
                np.array(all_targets), np.array(all_preds)
            )
            avg_metric = metrics.r2
        
        return avg_loss, avg_metric
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model on data.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
        
        Returns:
            Tuple of (y_true, y_pred, y_prob)
        """
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        for batch in dataloader:
            # Handle different batch formats
            if len(batch) == 2:
                X, y = batch
                mask = None
            elif len(batch) == 3:
                X, mask, y = batch
            else:
                raise ValueError(f"Unexpected batch format")
            
            # Move to device
            if isinstance(X, dict):
                X = {k: v.to(self.device) for k, v in X.items()}
            else:
                X = X.to(self.device)
            
            if mask is not None:
                mask = mask.to(self.device)
            
            y = y.to(self.device)
            
            # Get predictions
            if mask is not None:
                probs = model.predict(X, mask)
            else:
                probs = model.predict(X)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).float().cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        return (
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
        )
    
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: Optional[int] = None,
        lr: Optional[float] = None,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        Full training loop with optional early stopping.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            n_epochs: Number of epochs
            lr: Learning rate
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history dict
        """
        training_config = self.config.get("training", {})
        n_epochs = n_epochs or training_config.get("n_epochs", 100)
        lr = lr or training_config.get("lr", 1e-3)
        weight_decay = training_config.get("weight_decay", 1e-4)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
        
        history = {
            "train_loss": [],
            "train_metric": [],
            "val_loss": [],
            "val_metric": [],
        }
        
        best_val_metric = -np.inf
        patience_counter = 0
        best_state = None
        
        for epoch in range(n_epochs):
            # Training
            train_loss, train_metric = self.train_epoch(
                model, train_loader, optimizer
            )
            history["train_loss"].append(train_loss)
            history["train_metric"].append(train_metric)
            
            # Validation
            if val_loader is not None:
                model.eval()
                y_true, y_pred, y_prob = self.evaluate(model, val_loader)
                
                task = self.config.get("task", "classification")
                if task == "classification":
                    val_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
                    val_metric = val_metrics.auroc
                else:
                    val_metrics = compute_regression_metrics(y_true, y_pred)
                    val_metric = val_metrics.r2
                
                history["val_metric"].append(val_metric)
                
                # Early stopping check
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                scheduler.step(val_metric)
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                val_str = f", val_metric={val_metric:.4f}" if val_loader else ""
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"train_loss={train_loss:.4f}, train_metric={train_metric:.4f}"
                    f"{val_str}"
                )
        
        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        history["best_val_metric"] = best_val_metric
        
        return history
    
    def save_results(self, results: Dict, filename: str = "results.json"):
        """Save results to JSON file."""
        filepath = self.run_dir / filename
        
        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        with open(filepath, "w") as f:
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"Saved results to {filepath}")
