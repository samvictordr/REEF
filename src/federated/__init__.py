# Federated module
"""NVFlare components for Vertical Federated Learning on coral multi-omics."""

from .nvflare_vfl_runner import (
    NVFlareVFLTrainer,
    VFLConfig,
    OmicSilo,
    OmicsEncoder,
    VFLFusionHead,
    load_coral_data,
    compute_domain_saliency,
)

from .domain_aware_vfl import (
    REEF,
    DomainAwareConfig,
)

__all__ = [
    "NVFlareVFLTrainer",
    "VFLConfig", 
    "OmicSilo",
    "OmicsEncoder",
    "VFLFusionHead",
    "load_coral_data",
    "compute_domain_saliency",
    "REEF",
    "DomainAwareConfig",
]
