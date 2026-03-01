# LASER-VFL Adaptation Changelog

This document transparently describes all modifications made to the LASER-VFL baseline when adapting it for coral multi-omics data in the extreme P≫N regime (N=13 samples, P≈90,000 features).

## Architectural Fidelity

**The core LASER architecture is PRESERVED:**

| Component | Original LASER | Our Adapter | Status |
|-----------|----------------|-------------|--------|
| Feature Extractor | `MLPFeatureExtractor` (Linear→ReLU→Linear→ReLU→Linear) | Imported from `LASER-VFL-master/models/mlp.py` | ✓ Identical |
| Fusion Model | `FusionModel` (mean aggregation + Linear classifier) | Imported from `LASER-VFL-master/models/model_utils.py` | ✓ Identical |
| Task Sampling | Powerset-based sampling per head | Same algorithm via `powerset_except_empty()` | ✓ Identical |
| Multi-head Output | One head per client, subset predictions | Same structure | ✓ Identical |

## Stabilization Modifications

The following modifications address numerical instability specific to the extreme P≫N regime:

### 1. Weight Initialization (StabilizedFusionModel)

**Modification:** Small weight initialization for fusion head classifier  
**Original:** PyTorch default (kaiming_uniform)  
**Modified:** Normal(mean=0, std=1e-3) + zero bias  
**Rationale:** Large initial logits → softmax saturation → gradient vanishing

```python
# Applied to FusionModel.classifier after instantiation
nn.init.normal_(self.core_fusion.classifier.weight, mean=0.0, std=1e-3)
nn.init.zeros_(self.core_fusion.classifier.bias)
```

### 2. Optional LayerNorm (StabilizedMLPFeatureExtractor)

**Modification:** Optional LayerNorm on feature extractor output  
**Original:** No normalization after final Linear layer  
**Modified:** LayerNorm(cut_dim) on output when `use_layernorm=True`  
**Rationale:** Stabilizes embedding magnitudes with small batch sizes (batch_size < 8)

```python
# Added as wrapper around original MLPFeatureExtractor
out = self.core_extractor(x)  # Original LASER
if self.use_layernorm:
    out = self.output_norm(out)  # Stabilization
```

### 3. Optimizer Settings

**Modification:** AdamW with lower learning rate  
**Original:** Configurable (typically Adam with higher LR)  
**Modified:** AdamW, lr=1e-4, weight_decay=1e-4  
**Rationale:** Prevents overshooting in high-dimensional landscape

### 4. Gradient Clipping

**Modification:** Gradient norm clipping  
**Original:** Not applied  
**Modified:** `clip_grad_norm_(model.parameters(), max_norm=1.0)`  
**Rationale:** Prevents exploding gradients common in P≫N

### 5. Early Stopping

**Modification:** Early stopping on validation loss  
**Original:** Fixed epochs  
**Modified:** Patience=10 epochs  
**Rationale:** Prevents overfitting with N=13 training samples

## What Is NOT Modified

| Aspect | Description |
|--------|-------------|
| **MLP Architecture** | 3-layer MLP with ReLU activations (identical structure) |
| **Fusion Strategy** | Mean aggregation of client embeddings |
| **Task Sampling** | LASER's powerset-based training (core innovation preserved) |
| **Loss Function** | CrossEntropyLoss |
| **Forward Pass Logic** | Same multi-head output structure |

## Import Verification

The adapter explicitly imports original LASER classes:

```python
# From benchmarks/adapters/laser_adapter.py
from models.mlp import MLPFeatureExtractor  # Original LASER
from models.model_utils import FusionModel, BaseLaserModel  # Original LASER
```

The stabilized wrappers contain the original classes as `core_extractor` and `core_fusion`:

```python
>>> type(extractor.core_extractor).__name__
'MLPFeatureExtractor'

>>> type(fusion.core_fusion).__name__
'FusionModel'
```

## Recommendation for Manuscript

Include in Methods section:

> "The LASER baseline uses the original MLPFeatureExtractor and FusionModel 
> architectures from [LASER-VFL citation]. We apply minimal stabilization 
> modifications to address numerical instability in the extreme P≫N regime:
> (1) small weight initialization (σ=1e-3) for fusion classifiers,
> (2) optional LayerNorm on feature extractor outputs, and
> (3) gradient clipping (max_norm=1.0). These modifications do not alter
> LASER's core powerset-based task sampling algorithm."

---

*Last updated: December 13, 2025*
