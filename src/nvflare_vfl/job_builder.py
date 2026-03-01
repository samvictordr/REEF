"""
NVFlare Job Builder for Coral VFL.

Creates NVFlare job configurations programmatically.
This simplifies the creation of VFL jobs for multi-omics experiments.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# NVFlare 2.7.1 uses FedJobConfig
from nvflare.job_config.fed_job_config import FedJobConfig


def build_vfl_job_config(
    job_name: str = "coral_vfl",
    num_rounds: int = 50,
    data_path: str = "",
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    fusion_method: str = "concat",
    patience: int = 10,
    enable_domain_selection: bool = False,
    domain_warmup_rounds: int = 20,
    domain_topk: Optional[Dict[str, int]] = None,
    output_dir: str = "./jobs",
) -> Dict[str, Any]:
    """
    Build configuration for an NVFlare VFL job.
    
    Returns a dictionary that can be saved as JSON job config.
    
    Args:
        job_name: Name for the job
        num_rounds: Number of training rounds
        data_path: Path to data directory
        embedding_dim: Dimension of silo embeddings
        hidden_dim: Hidden layer dimension
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        fusion_method: How to fuse embeddings (concat, attention, gated)
        patience: Early stopping patience
        enable_domain_selection: Whether to enable domain feature selection
        domain_warmup_rounds: Rounds before gradient saliency selection
        domain_topk: Top-K features per omics type
        output_dir: Directory to save job config
    
    Returns:
        Dictionary containing job configuration
    """
    
    # Default domain saliency topk values
    if domain_topk is None:
        domain_topk = {
            "transcriptomics": 150,
            "proteomics": 250,
            "metabolomics": 500,
            "microbiome": 400,
        }
    
    # Client names for our 4 silos
    client_names = [
        "site-transcriptomics",
        "site-proteomics",
        "site-metabolomics",
        "site-microbiome",
    ]
    
    # Build job configuration
    job_config = {
        "name": job_name,
        "min_clients": 4,
        "num_rounds": num_rounds,
        "clients": client_names,
        "model_config": {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": 0.3,
            "fusion_method": fusion_method,
        },
        "training_config": {
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "grad_clip": 1.0,
        },
        "data_config": {
            "data_path": data_path,
            "batch_size": 8,
        },
        "domain_config": {
            "enable": enable_domain_selection,
            "warmup_rounds": domain_warmup_rounds,
            "topk": domain_topk,
        },
    }
    
    # Save to output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config_file = output_path / f"{job_name}_config.json"
        with open(config_file, "w") as f:
            json.dump(job_config, f, indent=2)
    
    return job_config


def build_vfl_job(
    job_name: str = "coral_vfl",
    num_rounds: int = 50,
    data_path: str = "",
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    fusion_method: str = "concat",
    patience: int = 10,
    enable_domain_selection: bool = False,
    domain_warmup_rounds: int = 20,
    domain_topk: Optional[Dict[str, int]] = None,
    output_dir: str = "./jobs",
) -> str:
    """
    Build an NVFlare VFL job.
    
    This is a wrapper that creates the job configuration and returns the path.
    """
    config = build_vfl_job_config(
        job_name=job_name,
        num_rounds=num_rounds,
        data_path=data_path,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        fusion_method=fusion_method,
        patience=patience,
        enable_domain_selection=enable_domain_selection,
        domain_warmup_rounds=domain_warmup_rounds,
        domain_topk=domain_topk,
        output_dir=output_dir,
    )
    
    output_path = Path(output_dir) / f"{job_name}_config.json"
    return str(output_path)


def build_domain_aware_vfl_job(
    job_name: str = "coral_vfl_domain",
    num_rounds: int = 60,
    data_path: str = "",
    domain_warmup_rounds: int = 20,
    domain_topk: Optional[Dict[str, int]] = None,
    output_dir: str = "./jobs",
) -> str:
    """
    Build a domain-aware VFL job.
    
    This is a convenience wrapper that enables domain feature selection
    with appropriate defaults.
    """
    
    return build_vfl_job(
        job_name=job_name,
        num_rounds=num_rounds,
        data_path=data_path,
        embedding_dim=64,
        hidden_dim=128,
        lr=5e-5,
        weight_decay=1e-4,
        fusion_method="concat",
        patience=10,
        enable_domain_selection=True,
        domain_warmup_rounds=domain_warmup_rounds,
        domain_topk=domain_topk,
        output_dir=output_dir,
    )
