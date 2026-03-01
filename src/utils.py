"""
Utility functions for coral VFL multi-omics project.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import numpy as np
import torch
import yaml


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.debug(f"Random seed set to {seed}")


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get torch device.
    
    Args:
        gpu_id: Specific GPU ID to use (None for auto-detect)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def load_yaml_config(config_path: Union[str, Path]) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded config from {config_path}")
    return config


def save_yaml_config(config: Dict, save_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.debug(f"Saved config to {save_path}")


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(
    base_dir: Union[str, Path],
    experiment_name: str,
    timestamp: bool = True,
) -> Path:
    """
    Create experiment output directory.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of experiment
        timestamp: Whether to add timestamp suffix
    
    Returns:
        Path to experiment directory
    """
    base_dir = Path(base_dir)
    
    if timestamp:
        dir_name = f"{experiment_name}_{get_timestamp()}"
    else:
        dir_name = experiment_name
    
    exp_dir = base_dir / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_number(n: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        n: Number to format
        precision: Decimal precision
    
    Returns:
        Formatted string
    """
    if abs(n) >= 1e9:
        return f"{n / 1e9:.{precision}f}B"
    elif abs(n) >= 1e6:
        return f"{n / 1e6:.{precision}f}M"
    elif abs(n) >= 1e3:
        return f"{n / 1e3:.{precision}f}K"
    else:
        return str(n)


def timer_decorator(func):
    """Decorator to time function execution."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result
    
    return wrapper


class EarlyStopping:
    """
    Early stopping helper.
    
    Stops training when a monitored metric has stopped improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        restore_best: bool = True,
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: "max" or "min" for the monitored metric
            restore_best: Whether to restore best model state on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_value = -np.inf if mode == "max" else np.inf
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None
        self.stopped = False
    
    def __call__(
        self,
        value: float,
        epoch: int,
        model: Optional[torch.nn.Module] = None,
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            epoch: Current epoch
            model: Model to save state
        
        Returns:
            True if training should stop
        """
        improved = False
        
        if self.mode == "max":
            if value > self.best_value + self.min_delta:
                improved = True
        else:
            if value < self.best_value - self.min_delta:
                improved = True
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            if model is not None and self.restore_best:
                self.best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stopped = True
            logger.info(
                f"Early stopping at epoch {epoch}. "
                f"Best: {self.best_value:.4f} at epoch {self.best_epoch}"
            )
            return True
        
        return False
    
    def restore(self, model: torch.nn.Module):
        """Restore best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.debug(f"Restored model state from epoch {self.best_epoch}")


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_git_info() -> Dict[str, str]:
    """
    Get git repository information.
    
    Returns:
        Dict with commit hash, branch, etc.
    """
    import subprocess
    
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        
        is_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        
        return {
            "commit": commit[:8],
            "branch": branch,
            "is_dirty": is_dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown", "is_dirty": False}


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dict with system specs
    """
    import platform
    
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    
    try:
        import nvflare
        info["nvflare_version"] = nvflare.__version__
    except ImportError:
        pass
    
    return info
