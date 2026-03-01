#!/usr/bin/env python3
"""
VFL vs LASER Benchmark: Compare NVFlare VFL+Domain Saliency against LASER baseline.

This script demonstrates that general VFL models struggle with small,
high-variance multi-omic datasets, while our VFL+Domain Saliency approach provides
better interpretability and handles data chaos more effectively.

Usage:
    # Quick benchmark (toy mode)
    python benchmarks/run_vfl_vs_laser.py --toy

    # Full benchmark with multiple seeds
    python benchmarks/run_vfl_vs_laser.py --seeds 42,123,456

    # With domain saliency analysis
    python benchmarks/run_vfl_vs_laser.py --compute-shap
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.nvflare_vfl_runner import (
    NVFlareVFLTrainer, VFLConfig, load_coral_data, compute_domain_saliency
)
from benchmarks.adapters.laser_adapter import LASERAdapter, load_coral_data as load_laser_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_nvflare_vfl(
    data_dir: Path,
    seed: int,
    n_rounds: int = 100,
    compute_shap: bool = False,
) -> Dict[str, Any]:
    """
    Run NVFlare VFL model.
    
    Returns:
        Results dict with metrics and timing
    """
    logger.info(f"Running NVFlare VFL (seed={seed})...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    # Load data
    silos, labels, splits = load_coral_data(data_dir, seed=seed)
    
    # Config
    config = VFLConfig(
        n_rounds=n_rounds,
        batch_size=8,
        lr_client=1e-3,
        lr_server=1e-3,
        seed=seed,
        patience=15,
    )
    
    # Train
    trainer = NVFlareVFLTrainer(silos, labels, config)
    history = trainer.train(splits["train"], splits["val"])
    
    # Evaluate
    test_metrics = trainer.evaluate(splits["test"], splits["train"])
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    results = {
        "method": "nvflare_vfl",
        "seed": seed,
        "test_auroc": test_metrics["auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "best_val_auroc": trainer.best_val_auroc,
        "n_rounds_completed": len(history["train_loss"]),
        "train_time_seconds": train_time,
    }
    
    # Optional domain saliency
    if compute_shap:
        domain_importance = compute_domain_saliency(trainer, splits["train"])
        results["shap_top_features"] = {}
        for omic, importance in domain_importance.items():
            top_idx = np.argsort(importance)[-10:][::-1]
            results["shap_top_features"][omic] = {
                "indices": top_idx.tolist(),
                "importance": importance[top_idx].tolist(),
            }
    
    logger.info(f"  NVFlare VFL AUROC: {test_metrics['auroc']:.4f}")
    
    return results


def run_laser(
    data_dir: Path,
    seed: int,
    n_epochs: int = 100,
) -> Dict[str, Any]:
    """
    Run LASER baseline.
    
    Returns:
        Results dict with metrics and timing
    """
    logger.info(f"Running LASER (seed={seed})...")
    set_seed(seed)
    
    start_time = datetime.now()
    
    try:
        # Load data using LASER's loader
        data, sample_ids = load_laser_data(data_dir / "processed")
        
        # Infer labels from sample IDs (HiT = stressed, Amb/Field = ambient)
        labels = np.array([1 if "HiT" in s else 0 for s in sample_ids])
        omic_names = list(data.keys())
        
        # Config matching NVFlare VFL for fair comparison
        config = {
            "hidden_dim": 128,
            "embedding_dim": 64,
            "n_epochs": n_epochs,
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "patience": 15,
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        adapter = LASERAdapter(
            data=data,
            labels=labels,
            omic_names=omic_names,
            config=config,
            device=device,
        )
        
        # Create splits
        n_samples = len(labels)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * n_samples)
        val_size = int(0.2 * n_samples)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        # Train
        metrics = adapter.train(train_idx, val_idx, n_epochs=n_epochs, batch_size=8)
        
        # Evaluate - create test dataloader
        test_loader = adapter.create_dataloader(test_idx, batch_size=8, shuffle=False, p_miss=0.0)
        test_metrics = adapter.evaluate(test_loader)
        
        train_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            "method": "laser",
            "seed": seed,
            "test_auroc": test_metrics.get("auroc", 0.5),
            "test_accuracy": test_metrics.get("accuracy", 0.5),
            "best_val_auroc": metrics.get("best_val_auroc", 0.5),
            "n_epochs_completed": metrics.get("epochs_completed", n_epochs),
            "train_time_seconds": train_time,
        }
        
        logger.info(f"  LASER AUROC: {results['test_auroc']:.4f}")
        
    except Exception as e:
        logger.error(f"LASER failed: {e}")
        results = {
            "method": "laser",
            "seed": seed,
            "test_auroc": 0.0,
            "test_accuracy": 0.0,
            "error": str(e),
        }
    
    return results


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    seeds: List[int],
    n_rounds: int = 100,
    compute_shap: bool = False,
) -> Dict[str, Any]:
    """
    Run full benchmark comparing NVFlare VFL vs LASER.
    
    Returns:
        Summary results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "nvflare_vfl": [],
        "laser": [],
    }
    
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed}")
        logger.info(f"{'='*60}")
        
        # Run NVFlare VFL
        vfl_results = run_nvflare_vfl(
            data_dir, seed, n_rounds, compute_shap
        )
        all_results["nvflare_vfl"].append(vfl_results)
        
        # Run LASER
        laser_results = run_laser(data_dir, seed, n_rounds)
        all_results["laser"].append(laser_results)
    
    # Aggregate results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "n_rounds": n_rounds,
        "methods": {},
    }
    
    for method in ["nvflare_vfl", "laser"]:
        aurocs = [r["test_auroc"] for r in all_results[method] if "error" not in r]
        accs = [r["test_accuracy"] for r in all_results[method] if "error" not in r]
        
        if aurocs:
            summary["methods"][method] = {
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "n_successful": len(aurocs),
            }
        else:
            summary["methods"][method] = {"error": "All runs failed"}
    
    # Print comparison
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*60}")
    
    for method, stats in summary["methods"].items():
        if "error" not in stats:
            logger.info(f"{method:15} | AUROC: {stats['mean_auroc']:.4f} ± {stats['std_auroc']:.4f}")
        else:
            logger.info(f"{method:15} | FAILED")
    
    # Determine winner
    vfl_stats = summary["methods"].get("nvflare_vfl", {})
    laser_stats = summary["methods"].get("laser", {})
    
    if "error" not in vfl_stats and "error" not in laser_stats:
        diff = vfl_stats["mean_auroc"] - laser_stats["mean_auroc"]
        if diff > 0.01:
            summary["conclusion"] = f"NVFlare VFL outperforms LASER by {diff:.4f} AUROC"
        elif diff < -0.01:
            summary["conclusion"] = f"LASER outperforms NVFlare VFL by {-diff:.4f} AUROC"
        else:
            summary["conclusion"] = "Methods perform similarly (within 0.01 AUROC)"
    
    # Save results
    summary["detailed_results"] = all_results
    
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NVFlare VFL+Domain Saliency vs LASER"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmark",
        help="Output directory"
    )
    parser.add_argument(
        "--seeds", type=str, default="42",
        help="Comma-separated seeds (e.g., 42,123,456)"
    )
    parser.add_argument(
        "--n-rounds", type=int, default=100,
        help="Training rounds/epochs"
    )
    parser.add_argument(
        "--toy", action="store_true",
        help="Quick toy run with fewer rounds"
    )
    parser.add_argument(
        "--compute-shap", action="store_true",
        help="Compute domain saliency feature importance"
    )
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(",")]
    
    # Paths
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    
    # Toy mode adjustments
    n_rounds = 20 if args.toy else args.n_rounds
    
    logger.info("VFL vs LASER Benchmark")
    logger.info(f"  Data: {data_dir}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Rounds: {n_rounds}")
    logger.info(f"  Domain Saliency: {args.compute_shap}")
    
    # Run benchmark
    summary = run_benchmark(
        data_dir=data_dir,
        output_dir=output_dir,
        seeds=seeds,
        n_rounds=n_rounds,
        compute_shap=args.compute_shap,
    )
    
    # Print conclusion
    if "conclusion" in summary:
        logger.info(f"\nCONCLUSION: {summary['conclusion']}")


if __name__ == "__main__":
    main()
