"""
SEAL model implementations for fragment-aware molecular property prediction.
"""

from .layers import SEALConv, SEALGINConv, GINConv
from .encoder import FragmentAwareEncoder, GINEncoder, load_encoder
from .seal import (
    TaskHead,
    MLPHead,
    MultiTaskModel,
    PretrainModel,
    RegressionModel,
    build_model,
)
from .random_forest import (
    RandomForestBaseline,
    train_rf_baseline,
)

__all__ = [
    # Layers
    "SEALConv",
    "SEALGINConv",
    "GINConv",
    # Encoders
    "FragmentAwareEncoder",
    "GINEncoder",
    "load_encoder",
    # Models
    "TaskHead",
    "MLPHead",
    "MultiTaskModel",
    "PretrainModel",
    "RegressionModel",
    "build_model",
    # Baselines
    "RandomForestBaseline",
    "train_rf_baseline",
]
