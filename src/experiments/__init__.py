# Experiments module
"""Experiment runners for coral multi-omics VFL research."""

from .base_experiment import BaseExperiment
from .exp_h1_vfl_vs_centralized import VFLvsCentralizedExperiment
from .exp_h2_mosaic_vs_complete import MosaicVsCompleteExperiment
from .experiment_h2_use_groups import UseGroupsExperiment

# H3 is deprecated - moved to experiments/deprecated/
# from .exp_h3_grouped_vs_ungrouped import GroupedVsUngroupedExperiment

__all__ = [
    "BaseExperiment",
    "VFLvsCentralizedExperiment",
    "MosaicVsCompleteExperiment",
    "UseGroupsExperiment",
]
