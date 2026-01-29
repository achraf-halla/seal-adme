"""
SEAL-ADME Models Module.

Provides fragment-aware graph neural networks for molecular property prediction:
- SEALConv / SEALGINConv: Message passing layers with intra/inter-fragment weights
- FragmentAwareGCNEncoder / FragmentAwareGINEncoder: Graph encoders
- MultiTaskPretrainModel: Classification pretraining
- MultiTaskRegressionModel: Regression finetuning
"""

from .layers import SEALConv, SEALGINConv

from .encoder import (
    FragmentAwareGCNEncoder,
    FragmentAwareGINEncoder,
    create_encoder,
    load_encoder,
)

from .seal import (
    MultiTaskPretrainModel,
    MultiTaskRegressionModel,
    build_pretrain_model,
    build_finetune_model,
)

__all__ = [
    # Layers
    "SEALConv",
    "SEALGINConv",
    # Encoders
    "FragmentAwareGCNEncoder",
    "FragmentAwareGINEncoder",
    "create_encoder",
    "load_encoder",
    # Models
    "MultiTaskPretrainModel",
    "MultiTaskRegressionModel",
    "build_pretrain_model",
    "build_finetune_model",
]
