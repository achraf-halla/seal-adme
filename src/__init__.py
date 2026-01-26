"""
SEAL-ADME: Learning Molecular Representations for ADME Prediction.

A framework for molecular property prediction using fragment-aware 
graph neural networks with built-in interpretability.

Modules:
    data: Data loading, preprocessing, and featurization
    models: SEAL model architectures (GCN and GIN variants)
    training: Training loops and utilities
    evaluation: Explanation extraction and visualization
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import training
from . import evaluation
