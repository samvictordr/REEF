"""
Tests for data loading and preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CoralMultiOmicsLoader


class TestCoralMultiOmicsLoader:
    """Tests for CoralMultiOmicsLoader."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with mock files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create mock omics data
        n_samples = 20
        sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]
        
        # Transcriptomics
        trans_df = pd.DataFrame(
            np.random.randn(n_samples, 100),
            index=sample_ids,
            columns=[f"gene_{i}" for i in range(100)],
        )
        trans_df.to_csv(Path(temp_dir) / "transcriptomics.csv")
        
        # Proteomics (fewer samples)
        prot_df = pd.DataFrame(
            np.random.randn(n_samples - 5, 50),
            index=sample_ids[:-5],
            columns=[f"protein_{i}" for i in range(50)],
        )
        prot_df.to_csv(Path(temp_dir) / "proteomics.csv")
        
        # Metabolomics
        metab_df = pd.DataFrame(
            np.random.randn(n_samples, 30),
            index=sample_ids,
            columns=[f"metabolite_{i}" for i in range(30)],
        )
        metab_df.to_csv(Path(temp_dir) / "metabolomics.csv")
        
        # Metadata
        metadata_df = pd.DataFrame({
            "sample_id": sample_ids,
            "bleached": np.random.randint(0, 2, n_samples),
            "color_score": np.random.uniform(1, 6, n_samples),
            "treatment": np.random.choice(["control", "stress"], n_samples),
        })
        metadata_df.to_csv(Path(temp_dir) / "metadata.csv", index=False)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_loader_initialization(self, temp_data_dir):
        """Test loader can be initialized."""
        loader = CoralMultiOmicsLoader(data_root=temp_data_dir)
        assert loader is not None
    
    def test_load_transcriptomics(self, temp_data_dir):
        """Test loading transcriptomics data."""
        loader = CoralMultiOmicsLoader(data_root=temp_data_dir)
        data = loader.load_omics("transcriptomics")
        
        assert data is not None
        assert data.shape[0] == 20
        assert data.shape[1] == 100
    
    def test_load_aligned_data(self, temp_data_dir):
        """Test loading and aligning multi-omics data."""
        loader = CoralMultiOmicsLoader(data_root=temp_data_dir)
        aligned = loader.load_all_aligned()
        
        # Should only keep samples present in all omics
        assert aligned is not None
        # Proteomics has 15 samples, so aligned should have <= 15
        assert aligned.n_samples <= 15
    
    def test_load_metadata(self, temp_data_dir):
        """Test loading metadata."""
        loader = CoralMultiOmicsLoader(data_root=temp_data_dir)
        labels = loader.load_labels(target_column="bleached")
        
        assert labels is not None
        assert len(labels) == 20


class TestPreprocessors:
    """Tests for data preprocessing."""
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        from src.data.preprocessors import OmicsPreprocessor
        
        data = np.random.randn(100, 50) * 10 + 5
        preprocessor = OmicsPreprocessor(normalization="zscore")
        
        normalized = preprocessor.fit_transform(data)
        
        # Mean should be ~0, std should be ~1
        assert np.abs(normalized.mean()) < 0.1
        assert np.abs(normalized.std() - 1) < 0.1
    
    def test_feature_selection(self):
        """Test variance-based feature selection."""
        from src.data.preprocessors import OmicsPreprocessor
        
        # Create data with some low-variance features
        data = np.random.randn(100, 50)
        data[:, 40:] = 1.0  # Low variance features
        
        preprocessor = OmicsPreprocessor(
            normalization=None,
            feature_selection="variance",
            n_features=30,
        )
        
        selected = preprocessor.fit_transform(data)
        
        assert selected.shape[1] == 30


class TestDatasets:
    """Tests for PyTorch datasets."""
    
    def test_coral_dataset_creation(self):
        """Test CoralMultiOmicsDataset creation."""
        from src.data.dataset import CoralMultiOmicsDataset
        
        # Create mock data
        omics_dict = {
            "transcriptomics": np.random.randn(50, 100).astype(np.float32),
            "proteomics": np.random.randn(50, 50).astype(np.float32),
        }
        labels = np.random.randint(0, 2, 50)
        
        dataset = CoralMultiOmicsDataset(omics_dict, labels)
        
        assert len(dataset) == 50
        
        # Get a sample
        X, y = dataset[0]
        assert "transcriptomics" in X
        assert "proteomics" in X
        assert y in [0, 1]
    
    def test_mosaic_dataset_with_missing(self):
        """Test MosaicMultiOmicsDataset with missing data."""
        from src.data.dataset import MosaicMultiOmicsDataset
        
        # Create mock data with some NaN
        omics_dict = {
            "transcriptomics": np.random.randn(50, 100).astype(np.float32),
            "proteomics": np.random.randn(50, 50).astype(np.float32),
        }
        labels = np.random.randint(0, 2, 50)
        
        # Add some missing
        availability = np.ones((50, 2), dtype=np.float32)
        availability[10:20, 1] = 0  # Missing proteomics for samples 10-19
        
        dataset = MosaicMultiOmicsDataset(
            omics_dict, 
            labels,
            availability=availability,
        )
        
        assert len(dataset) == 50
        
        # Check missing sample
        X, mask, y = dataset[15]
        assert mask[1] == 0  # Proteomics should be masked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
