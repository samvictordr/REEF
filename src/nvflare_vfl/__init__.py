"""                                                                                           
NVFlare-based Vertical Federated Learning for Coral Multi-Omics.                              
                                                                                               
This module provides real NVFlare implementation for VFL experiments,                         
not simulated federated learning. It uses the official NVFlare APIs:                          
                                                                                               
- Executor: Base class for client-side computation                                            
- Controller: Base class for server-side orchestration                                          
- DXO: Data Exchange Object for secure communication                                          
- Simulator: Single-machine testing of distributed jobs                                       
                                                                                               
Components:                                                                                    
-----------                                                                                    
- OmicsEncoder: Neural network encoder for each omics silo                                    
- VFLFusionHead: Server-side fusion network                                                   
- CoralSiloExecutor: NVFlare Executor for client silos                                        
- CoralVFLController: NVFlare Controller for server orchestration                             
- CoralVFLSimulator: Wrapper for NVFlare simulator                                            
- NVFlareNativeVFLTrainer: VFL trainer using real NVFlare APIs                                
- DomainAwareNVFlareTrainer: Domain-aware VFL with NVFlare APIs                                
- build_vfl_job: Job builder using NVFlare Job API                                            
- run_vfl_experiment: High-level experiment runner                                            
"""

from .models import OmicsEncoder, VFLFusionHead
from .silo_executor import CoralSiloExecutor
from .vfl_controller import CoralVFLController
from .job_builder import build_vfl_job, build_domain_aware_vfl_job
from .simulator_runner import CoralVFLSimulator, run_vfl_experiment, run_cross_validation
from .nvflare_native_trainer import (
    NVFlareNativeVFLTrainer,
    NVFlareVFLConfig,
    DomainAwareNVFlareTrainer,
    DomainAwareNVFlareConfig,
    load_coral_data,
    NVFlareDXOManager,
)

__all__ = [
    # Neural network models
    "OmicsEncoder",
    "VFLFusionHead",
    # NVFlare components
    "CoralSiloExecutor",
    "CoralVFLController",
    # Job building and running
    "build_vfl_job",
    "build_domain_aware_vfl_job",
    "CoralVFLSimulator",
    "run_vfl_experiment",
    "run_cross_validation",
    # NVFlare-native trainers
    "NVFlareNativeVFLTrainer",
    "NVFlareVFLConfig",
    "DomainAwareNVFlareTrainer",
    "DomainAwareNVFlareConfig",
    "load_coral_data",
    "NVFlareDXOManager",
]
