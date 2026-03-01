"""
Tests for VFL models.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    SingleOmicsModel,
    EarlyFusionModel,
    LateFusionModel,
)
from src.models.vertical_federated import VFLClient, VFLServer


class TestCentralizedModels:
    """Tests for centralized baseline models."""
    
    def test_single_omics_model(self):
        """Test SingleOmicsModel forward pass."""
        model = SingleOmicsModel(
            input_dim=100,
            hidden_dims=[64, 32],
            n_classes=2,
        )
        
        x = torch.randn(16, 100)
        logits = model(x)
        
        assert logits.shape == (16, 1)  # Binary classification
    
    def test_early_fusion_model(self):
        """Test EarlyFusionModel with multiple omics."""
        input_dims = {"transcriptomics": 100, "proteomics": 50, "metabolomics": 30}
        
        model = EarlyFusionModel(
            input_dims=input_dims,
            hidden_dims=[128, 64],
            n_classes=2,
        )
        
        x = {
            "transcriptomics": torch.randn(16, 100),
            "proteomics": torch.randn(16, 50),
            "metabolomics": torch.randn(16, 30),
        }
        
        logits = model(x)
        
        assert logits.shape == (16, 1)
    
    def test_late_fusion_model(self):
        """Test LateFusionModel with attention fusion."""
        input_dims = {"transcriptomics": 100, "proteomics": 50}
        
        model = LateFusionModel(
            input_dims=input_dims,
            encoder_hidden_dim=64,
            embedding_dim=32,
            fusion_hidden_dim=64,
            n_classes=2,
            fusion_method="attention",
        )
        
        x = {
            "transcriptomics": torch.randn(16, 100),
            "proteomics": torch.randn(16, 50),
        }
        
        logits = model(x)
        
        assert logits.shape == (16, 1)


class TestVFLModels:
    """Tests for VFL models."""
    
    def test_vfl_client(self):
        """Test VFLClient encoder."""
        client = VFLClient(
            input_dim=100,
            hidden_dim=64,
            embedding_dim=32,
        )
        
        x = torch.randn(16, 100)
        embedding = client(x)
        
        assert embedding.shape == (16, 32)
    
    def test_vfl_server(self):
        """Test VFLServer fusion head."""
        server = VFLServer(
            n_clients=4,
            embedding_dim=32,
            hidden_dim=64,
            n_classes=2,
        )
        
        # Simulate embeddings from 4 clients
        embeddings = [torch.randn(16, 32) for _ in range(4)]
        
        logits = server(embeddings)
        
        assert logits.shape == (16, 1)
    
    def test_vfl_with_mask(self):
        """Test VFL with availability mask."""
        server = VFLServer(
            n_clients=4,
            embedding_dim=32,
            hidden_dim=64,
            n_classes=2,
            fusion_method="masked_concat",
        )
        
        embeddings = [torch.randn(16, 32) for _ in range(4)]
        mask = torch.ones(16, 4)
        mask[:8, 2] = 0  # Client 2 missing for first 8 samples
        
        logits = server(embeddings, mask)
        
        assert logits.shape == (16, 1)
    
    def test_vfl_end_to_end(self):
        """Test VFL clients + server end-to-end."""
        input_dims = [100, 50, 30, 20]  # 4 silos
        embedding_dim = 32
        
        # Create clients
        clients = [
            VFLClient(dim, hidden_dim=64, embedding_dim=embedding_dim)
            for dim in input_dims
        ]
        
        # Create server
        server = VFLServer(
            n_clients=4,
            embedding_dim=embedding_dim,
            hidden_dim=64,
            n_classes=2,
        )
        
        # Forward pass
        batch_size = 16
        inputs = [torch.randn(batch_size, dim) for dim in input_dims]
        
        embeddings = [client(x) for client, x in zip(clients, inputs)]
        logits = server(embeddings)
        
        # Backward pass
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(),
            torch.randint(0, 2, (batch_size,)).float(),
        )
        loss.backward()
        
        # Check gradients flow back to all clients
        for i, client in enumerate(clients):
            for param in client.parameters():
                assert param.grad is not None, f"Client {i} has no gradients"


class TestModelUtils:
    """Tests for model utilities."""
    
    def test_parameter_count(self):
        """Test parameter counting."""
        from src.utils import count_parameters
        
        model = SingleOmicsModel(
            input_dim=100,
            hidden_dims=[64, 32],
            n_classes=2,
        )
        
        n_params = count_parameters(model)
        
        # Should be > 0
        assert n_params > 0
        
        # Rough check: 100*64 + 64*32 + 32*1 = 6400 + 2048 + 32 = 8480
        # Plus biases and batch norm params
        assert n_params > 8000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
