"""
SEAL-ADME Models Module.

This module provides fragment-aware graph neural network architectures
for molecular property prediction with built-in interpretability.

Layers:
    SEALConv: Fragment-aware GCN convolution
    SEALGINConv: Fragment-aware GIN convolution

Encoders:
    GCNEncoder: GCN-based fragment encoder
    GINEncoder: GIN-based fragment encoder
    FragmentAwareEncoder: Alias for GCNEncoder
    SEALGINEncoder: Alias for GINEncoder

Models:
    MultiTaskModel: Multi-task prediction model
    PretrainModel: Model for classification pretraining
    FinetuneModel: Model for regression finetuning

Factory Functions:
    create_encoder: Create encoder by type
    create_model: Create complete model
    load_pretrained_encoder: Load encoder from checkpoint
"""

from .layers import (
    SEALConv,
    SEALGINConv,
)

from .encoders import (
    BaseEncoder,
    GCNEncoder,
    GINEncoder,
    FragmentAwareEncoder,
    SEALGINEncoder,
    create_encoder,
)

from .heads import (
    TaskHead,
    MultiTaskModel,
    PretrainModel,
    FinetuneModel,
    create_model,
    load_pretrained_encoder,
)


__all__ = [
    # Layers
    "SEALConv",
    "SEALGINConv",
    # Encoders
    "BaseEncoder",
    "GCNEncoder",
    "GINEncoder",
    "FragmentAwareEncoder",
    "SEALGINEncoder",
    "create_encoder",
    # Models
    "TaskHead",
    "MultiTaskModel",
    "PretrainModel",
    "FinetuneModel",
    "create_model",
    "load_pretrained_encoder",
]
