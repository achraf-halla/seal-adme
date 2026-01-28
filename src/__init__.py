"""
SEAL-ADME: Substructure-aware Explainable ADME Learning.

A fragment-aware graph neural network framework for molecular property
prediction with built-in interpretability.

Submodules:
    data: Data loading, preprocessing, and featurization
    models: GNN architectures (GCN, GIN encoders)
    training: Training loops and metrics
    explanations: Explanation extraction and visualization
"""

__version__ = "0.1.0"

# Note: Submodules are not auto-imported to avoid circular dependencies.
# Import them explicitly:
#   from src.data import ...
#   from src.models import ...
#   from src.training import ...
#   from src.explanations import ...
