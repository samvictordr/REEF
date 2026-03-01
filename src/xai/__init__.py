# XAI Module for Coral Multi-Omics
"""
Explainable AI utilities for inferring missing omic representations.

This module provides Domain-aware inference pipelines to generate proxy
representations for missing omics blocks in the VFL workflow.

Includes:
- Model wrappers for predict_proba interface
- XAI pipeline for permutation importance + domain saliency + surrogate models
- Domain helpers for computing and aggregating saliency values
"""

from .infer_missing_omic import MissingOmicInferrer, train_inference_model
from .domain_helpers import (
    DomainExplainerWrapper,
    compute_domain_saliency_values,
    aggregate_saliency_to_latent,
    compute_inference_fidelity,
)
from .model_wrappers import (
    MosaicPredictWrapper,
    SklearnPredictWrapper,
    SurrogateWrapper,
    create_predict_wrapper,
    set_seeds,
    validate_model_output,
)
from .mosaic_xai_pipeline import (
    XAIPipelineConfig,
    run_xai_pipeline,
    compute_permutation_importance,
    compute_shap_values as compute_shap_values_pipeline,
    summarize_shap_values,
    train_surrogate_lasso,
)

__all__ = [
    # Missing omic inference
    "MissingOmicInferrer",
    "train_inference_model",
    # Domain saliency helpers
    "DomainExplainerWrapper", 
    "compute_domain_saliency_values",
    "aggregate_saliency_to_latent",
    "compute_inference_fidelity",
    # Model wrappers
    "MosaicPredictWrapper",
    "SklearnPredictWrapper",
    "SurrogateWrapper",
    "create_predict_wrapper",
    "set_seeds",
    "validate_model_output",
    # XAI pipeline
    "XAIPipelineConfig",
    "run_xai_pipeline",
    "compute_permutation_importance",
    "summarize_shap_values",
    "train_surrogate_lasso",
]
