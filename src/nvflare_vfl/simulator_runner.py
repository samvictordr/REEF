"""
NVFlare Simulator Runner for Coral VFL.

This module provides functions to run NVFlare VFL jobs using the
NVFlare Simulator API, which is the standard way to test and run
federated learning jobs in a single-machine environment.

The simulator accurately replicates the behavior of a distributed
NVFlare deployment, including proper message passing and FL context.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import torch

# NVFlare imports
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner
from nvflare.fuel.utils.config_service import ConfigService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoralVFLSimulator:
    """
    Runs Coral VFL jobs using NVFlare Simulator.
    
    This class wraps the NVFlare simulator to provide a simple interface
    for running VFL experiments on coral multi-omics data.
    """
    
    def __init__(
        self,
        job_folder: str,
        workspace: str = "",
        n_clients: int = 4,
        threads: int = 4,
        gpu: Optional[str] = None,
    ):
        """
        Initialize the simulator.
        
        Args:
            job_folder: Path to the NVFlare job folder
            workspace: Working directory for simulator
            n_clients: Number of clients (should be 4 for 4 omics types)
            threads: Number of threads per client
            gpu: GPU device IDs to use (e.g., "0,1")
        """
        self.job_folder = Path(job_folder)
        self.n_clients = n_clients
        self.threads = threads
        self.gpu = gpu
        
        # Setup workspace
        if workspace:
            self.workspace = Path(workspace)
        else:
            self.workspace = Path(tempfile.mkdtemp(prefix="nvflare_coral_"))
        
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Client names for our 4 silos
        self.client_names = [
            "site-transcriptomics",
            "site-proteomics",
            "site-metabolomics",
            "site-microbiome",
        ]
        
        logger.info(f"CoralVFLSimulator initialized")
        logger.info(f"  Job folder: {self.job_folder}")
        logger.info(f"  Workspace: {self.workspace}")
        logger.info(f"  Clients: {self.client_names}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the VFL simulation.
        
        Returns:
            Dictionary with results including metrics and trained model info
        """
        logger.info("Starting VFL simulation...")
        
        # Create simulator runner
        runner = SimulatorRunner(
            job_folder=str(self.job_folder),
            workspace=str(self.workspace),
            clients=",".join(self.client_names),
            n_clients=self.n_clients,
            threads=self.threads,
            gpu=self.gpu or "",
        )
        
        # Run the simulation
        start_time = datetime.now()
        return_code = runner.run()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        if return_code != 0:
            logger.error(f"Simulation failed with return code {return_code}")
            return {
                "success": False,
                "return_code": return_code,
                "duration_seconds": duration,
            }
        
        logger.info(f"Simulation completed in {duration:.2f} seconds")
        
        # Collect results
        results = self._collect_results()
        results["success"] = True
        results["return_code"] = return_code
        results["duration_seconds"] = duration
        
        return results
    
    def _collect_results(self) -> Dict[str, Any]:
        """Collect results from the simulation workspace."""
        results = {
            "workspace": str(self.workspace),
            "metrics": {},
            "model_path": None,
        }
        
        # Look for saved metrics
        metrics_dir = self.workspace / "simulate_job" / "server"
        if metrics_dir.exists():
            for f in metrics_dir.glob("*.json"):
                try:
                    with open(f) as fp:
                        results["metrics"][f.stem] = json.load(fp)
                except:
                    pass
        
        # Look for saved model
        model_dir = self.workspace / "simulate_job" / "server" / "models"
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            if model_files:
                results["model_path"] = str(model_files[0])
        
        return results
    
    def cleanup(self):
        """Clean up temporary workspace."""
        if self.workspace.exists() and "tmp" in str(self.workspace):
            shutil.rmtree(self.workspace)
            logger.info(f"Cleaned up workspace: {self.workspace}")


def run_vfl_experiment(
    data_path: str,
    output_dir: str = "outputs/nvflare",
    num_rounds: int = 50,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    enable_domain_selection: bool = False,
    domain_warmup_rounds: int = 20,
    domain_topk: Optional[Dict[str, int]] = None,
    n_threads: int = 4,
    gpu: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a VFL experiment using NVFlare simulator.
    
    This is the main entry point for running coral VFL experiments.
    
    Args:
        data_path: Path to directory containing omics data files
        output_dir: Directory to save outputs
        num_rounds: Number of training rounds
        embedding_dim: Embedding dimension
        hidden_dim: Hidden layer dimension  
        lr: Learning rate
        weight_decay: Weight decay
        enable_domain_selection: Whether to use gradient saliency feature selection
        domain_warmup_rounds: Warmup rounds before gradient saliency
        domain_topk: Top-K features per omics
        n_threads: Number of threads
        gpu: GPU device string
    
    Returns:
        Dictionary with experiment results
    """
    from .job_builder import build_vfl_job
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build the job
    job_name = "coral_vfl_domain" if enable_domain_selection else "coral_vfl"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name_full = f"{job_name}_{timestamp}"
    
    logger.info(f"Building NVFlare job: {job_name_full}")
    
    job_path = build_vfl_job(
        job_name=job_name_full,
        num_rounds=num_rounds,
        data_path=data_path,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        fusion_method="concat",
        patience=10,
        enable_domain_selection=enable_domain_selection,
        domain_warmup_rounds=domain_warmup_rounds,
        domain_topk=domain_topk,
        output_dir=str(output_path / "jobs"),
    )
    
    logger.info(f"Job built at: {job_path}")
    
    # Create simulator
    workspace = str(output_path / "workspace" / job_name_full)
    simulator = CoralVFLSimulator(
        job_folder=job_path,
        workspace=workspace,
        n_clients=4,
        threads=n_threads,
        gpu=gpu,
    )
    
    # Run simulation
    results = simulator.run()
    
    # Save results
    results_file = output_path / f"{job_name_full}_results.json"
    with open(results_file, "w") as f:
        # Make results JSON serializable
        serializable = {k: v for k, v in results.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        json.dump(serializable, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return results


def run_cross_validation(
    data_path: str,
    output_dir: str = "outputs/nvflare_cv",
    n_folds: int = 5,
    num_rounds: int = 50,
    enable_domain_selection: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run cross-validated VFL experiment.
    
    Args:
        data_path: Path to data directory
        output_dir: Output directory
        n_folds: Number of CV folds
        num_rounds: Rounds per fold
        enable_domain_selection: Use gradient saliency selection
        **kwargs: Additional arguments for run_vfl_experiment
    
    Returns:
        Dictionary with CV results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    
    for fold in range(n_folds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Running fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*50}\n")
        
        fold_output = str(output_path / f"fold_{fold}")
        
        result = run_vfl_experiment(
            data_path=data_path,
            output_dir=fold_output,
            num_rounds=num_rounds,
            enable_domain_selection=enable_domain_selection,
            **kwargs
        )
        
        fold_results.append(result)
    
    # Aggregate results
    cv_results = {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "enable_domain_selection": enable_domain_selection,
    }
    
    # Compute mean/std metrics if available
    if all(r.get("success", False) for r in fold_results):
        cv_results["all_successful"] = True
        
        # Extract metrics from each fold if available
        aurocs = []
        for r in fold_results:
            if "metrics" in r and "auroc" in r["metrics"]:
                aurocs.append(r["metrics"]["auroc"])
        
        if aurocs:
            cv_results["mean_auroc"] = float(np.mean(aurocs))
            cv_results["std_auroc"] = float(np.std(aurocs))
    else:
        cv_results["all_successful"] = False
    
    # Save CV results
    cv_file = output_path / "cv_results.json"
    with open(cv_file, "w") as f:
        json.dump(cv_results, f, indent=2)
    
    logger.info(f"\nCV results saved to: {cv_file}")
    
    return cv_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Coral VFL with NVFlare")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/nvflare", help="Output directory")
    parser.add_argument("--num-rounds", type=int, default=50, help="Training rounds")
    parser.add_argument("--enable-shap", action="store_true", help="Enable gradient saliency selection")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device IDs")
    parser.add_argument("--threads", type=int, default=4, help="Threads per client")
    
    args = parser.parse_args()
    
    if args.cv:
        results = run_cross_validation(
            data_path=args.data_path,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            num_rounds=args.num_rounds,
            enable_domain_selection=args.enable_domain_selection,
            n_threads=args.threads,
            gpu=args.gpu,
        )
    else:
        results = run_vfl_experiment(
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_rounds=args.num_rounds,
            enable_domain_selection=args.enable_domain_selection,
            n_threads=args.threads,
            gpu=args.gpu,
        )
    
    print("\nExperiment completed!")
    print(f"Success: {results.get('success', results.get('all_successful', False))}")
