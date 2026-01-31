"""
SEAL-ADME Data Processing Module.

Simplified pipeline for:
- Loading ADME datasets from TDC
- Graph featurization with BRICS decomposition
- Normalization for regression tasks
"""

from .constants import (
    PRETRAIN_TASKS,
    FINETUNE_TASKS,
    ATOM_TYPE_SET,
    HYBRIDIZATION_SET,
    HALOGEN_ATOMIC_NUMS,
    PAULING_ELECTRONEGATIVITY,
    DEFAULT_ELECTRONEGATIVITY,
    STANDARD_COLUMNS,
    META_COLUMNS,
    DEFAULT_SPLIT_FRACTIONS,
    DEFAULT_INPUT_FEATURES,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT,
    AURORA_SEARCH_TERMS,
)

from .loaders import (
    TDCLoader,
    load_pretrain_data,
    load_finetune_data,
)

from .graph_featurizer import (
    GraphFeaturizer,
    FragmentExtractor,
    save_graphs,
    load_graphs,
)

__all__ = [
    # Constants
    "PRETRAIN_TASKS",
    "FINETUNE_TASKS",
    "ATOM_TYPE_SET",
    "HYBRIDIZATION_SET",
    "HALOGEN_ATOMIC_NUMS",
    "PAULING_ELECTRONEGATIVITY",
    "DEFAULT_ELECTRONEGATIVITY",
    "STANDARD_COLUMNS",
    "META_COLUMNS",
    "DEFAULT_SPLIT_FRACTIONS",
    "DEFAULT_INPUT_FEATURES",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_DROPOUT",
    "AURORA_SEARCH_TERMS",
    # Loaders
    "TDCLoader",
    "load_pretrain_data",
    "load_finetune_data",
    # Graph featurization
    "GraphFeaturizer",
    "FragmentExtractor",
    "save_graphs",
    "load_graphs",
]
