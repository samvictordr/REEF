"""
NVFlare Data Loader Component for Coral Multi-omics.

Loads coral multi-omics data for each silo.
"""

import os
from typing import Optional
import logging

import numpy as np
import pandas as pd
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


logger = logging.getLogger(__name__)


class CoralDataLoader(FLComponent):
    """
    Data loader for coral multi-omics data.
    
    Each site loads its assigned omics type from the data directory.
    """
    
    def __init__(self, data_root: str = "/data"):
        super().__init__()
        self.data_root = data_root
        self._cache = {}
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle FL events."""
        pass
    
    def load_omics(self, omics_type: str) -> np.ndarray:
        """
        Load omics data.
        
        Args:
            omics_type: Type of omics (transcriptomics, proteomics, etc.)
        
        Returns:
            Numpy array of shape (n_samples, n_features)
        """
        if omics_type in self._cache:
            return self._cache[omics_type]
        
        # Define file patterns
        file_patterns = {
            "transcriptomics": "transcriptomics*.csv",
            "proteomics": "proteomics*.csv",
            "metabolomics": "metabolomics*.csv",
            "microbiome": "microbiome*.csv",
        }
        
        pattern = file_patterns.get(omics_type)
        if pattern is None:
            raise ValueError(f"Unknown omics type: {omics_type}")
        
        # Search for file
        import glob
        files = glob.glob(os.path.join(self.data_root, "**", pattern), recursive=True)
        
        if not files:
            # Try alternative naming
            alt_patterns = {
                "transcriptomics": ["gene_expression*.csv", "rnaseq*.csv", "host_transcriptome*.csv"],
                "proteomics": ["protein*.csv"],
                "metabolomics": ["metabolite*.csv"],
                "microbiome": ["16S*.csv", "microbiota*.csv", "symbiodinium*.csv"],
            }
            
            for alt_pattern in alt_patterns.get(omics_type, []):
                files = glob.glob(os.path.join(self.data_root, "**", alt_pattern), recursive=True)
                if files:
                    break
        
        if not files:
            logger.warning(f"No data file found for {omics_type}, generating placeholder")
            # Generate placeholder data for testing
            data = np.random.randn(50, 1000).astype(np.float32)
        else:
            # Load first matching file
            file_path = files[0]
            logger.info(f"Loading {omics_type} from {file_path}")
            
            df = pd.read_csv(file_path, index_col=0)
            data = df.values.astype(np.float32)
        
        self._cache[omics_type] = data
        return data
    
    def load_labels(self) -> np.ndarray:
        """Load target labels."""
        # Search for metadata file
        import glob
        files = glob.glob(os.path.join(self.data_root, "**/metadata*.csv"), recursive=True)
        
        if not files:
            files = glob.glob(os.path.join(self.data_root, "**/sample_info*.csv"), recursive=True)
        
        if not files:
            logger.warning("No metadata file found, generating placeholder labels")
            return np.random.randint(0, 2, size=50)
        
        df = pd.read_csv(files[0])
        
        # Try to find bleaching/stress column
        for col in ["bleached", "bleaching", "stress", "condition", "treatment", "status"]:
            if col in df.columns:
                labels = df[col].values
                # Convert to binary if needed
                if labels.dtype == object:
                    unique = np.unique(labels)
                    labels = (labels == unique[0]).astype(int)
                return labels
        
        logger.warning("No target column found in metadata")
        return np.random.randint(0, 2, size=len(df))
    
    def get_sample_indices(
        self, 
        split: str = "train",
        fold: int = 0,
    ) -> np.ndarray:
        """
        Get sample indices for a specific split and fold.
        
        Args:
            split: train, val, or test
            fold: CV fold number
        
        Returns:
            Array of sample indices
        """
        # Load splits file if exists
        splits_file = os.path.join(self.data_root, "splits", f"fold_{fold}.npz")
        
        if os.path.exists(splits_file):
            data = np.load(splits_file)
            return data[f"{split}_indices"]
        else:
            # Generate random split
            n_samples = 50
            indices = np.arange(n_samples)
            np.random.seed(42 + fold)
            np.random.shuffle(indices)
            
            train_end = int(0.7 * n_samples)
            val_end = int(0.85 * n_samples)
            
            if split == "train":
                return indices[:train_end]
            elif split == "val":
                return indices[train_end:val_end]
            else:
                return indices[val_end:]
