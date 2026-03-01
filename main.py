#!/usr/bin/env python
"""
Main entry point for coral VFL multi-omics experiments.

Usage:
    python main.py --experiment h1 --config config/exp_h1_vfl_vs_centralized.yml
    python main.py --experiment h2 --config config/exp_h2_mosaic_vs_complete.yml
    python main.py --experiment h3 --config config/exp_h3_grouped_vs_ungrouped.yml
    python main.py --experiment all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(experiment_name: str, config_path: Path, output_dir: Path):
    """Run a single experiment."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment: {experiment_name}")
    
    # Import experiment class based on name
    if experiment_name == "h1":
        from src.experiments import VFLvsCentralizedExperiment as ExperimentClass
    elif experiment_name == "h2":
        from src.experiments import MosaicExperiment as ExperimentClass
    elif experiment_name == "h3":
        from src.experiments import GroupedVsUngroupedExperiment as ExperimentClass
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    # Load config
    config = load_config(config_path)
    
    # Create and run experiment
    experiment = ExperimentClass(
        config=config,
        output_dir=output_dir,
    )
    
    results = experiment.run()
    
    logger.info(f"Experiment {experiment_name} completed successfully")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run coral VFL multi-omics experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --experiment h1
    python main.py --experiment h2 --fold 0
    python main.py --experiment all
        """,
    )
    
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        choices=["h1", "h2", "h3", "all"],
        help="Experiment to run (h1, h2, h3, or all)",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config file (optional, uses default for experiment)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only specific fold (0-4)",
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seeds
    import random
    import numpy as np
    import torch
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Default configs
    default_configs = {
        "h1": Path("config/exp_h1_vfl_vs_centralized.yml"),
        "h2": Path("config/exp_h2_mosaic_vs_complete.yml"),
        "h3": Path("config/exp_h3_grouped_vs_ungrouped.yml"),
    }
    
    # Run experiments
    experiments_to_run = ["h1", "h2", "h3"] if args.experiment == "all" else [args.experiment]
    
    all_results = {}
    
    for exp_name in experiments_to_run:
        # Determine config path
        if args.config and args.experiment != "all":
            config_path = args.config
        else:
            config_path = default_configs[exp_name]
        
        # Check config exists
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            continue
        
        # Determine output directory
        output_dir = args.output / f"h{exp_name[-1]}_results" if "h" not in exp_name else args.output / f"{exp_name}_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = run_experiment(exp_name, config_path, output_dir)
            all_results[exp_name] = results
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    for exp_name, results in all_results.items():
        logger.info(f"\n{exp_name.upper()}:")
        if isinstance(results, dict) and "best_model" in results:
            logger.info(f"  Best model: {results['best_model']}")
        if isinstance(results, dict) and "metrics" in results:
            metrics = results["metrics"]
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    

if __name__ == "__main__":
    main()
