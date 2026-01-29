"""
SEAL-ADME Data Processing Module.

This module provides utilities for:
- Loading data from TDC and ChEMBL
- SMILES validation and standardization
- Scaffold-based data splitting
- BRICS molecular fragmentation
- PyG graph construction
"""

from .constants import (
    PRETRAIN_TASKS,
    FINETUNE_TASKS,
    ATOM_TYPE_SET,
    HYBRIDIZATION_SET,
    PAULING_ELECTRONEGATIVITY,
    DEFAULT_ELECTRONEGATIVITY,
    HALOGEN_ATOMIC_NUMS,
    META_COLUMNS,
    STANDARD_COLUMNS,
    AURORA_SEARCH_TERMS,
    DEFAULT_SPLIT_FRACTIONS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT,
    DEFAULT_INPUT_FEATURES,
)

from .loaders import (
    TDCLoader,
    ChEMBLAuroraLoader,
    filter_aurora_data,
    convert_aurora_to_standard,
    infer_task_type,
)

from .preprocessing import (
    canonicalize_smiles,
    validate_smiles_column,
    deduplicate_by_label_consistency,
    standardize_dataframe,
    DataPreprocessor,
)

from .splitting import (
    create_scaffold_split,
    create_random_split,
    split_by_task,
)

from .fragmentation import (
    brics_decompose,
    get_fragment_membership_matrix,
    get_edge_break_mask,
    FragmentExtractor,
)

from .graph_builder import (
    one_hot_encode,
    get_atom_features,
    mol_to_graph,
    GraphBuilder,
    save_graphs,
    load_graphs,
)

__all__ = [
    # Constants
    "PRETRAIN_TASKS",
    "FINETUNE_TASKS",
    "ATOM_TYPE_SET",
    "HYBRIDIZATION_SET",
    "PAULING_ELECTRONEGATIVITY",
    "DEFAULT_ELECTRONEGATIVITY",
    "HALOGEN_ATOMIC_NUMS",
    "META_COLUMNS",
    "STANDARD_COLUMNS",
    "AURORA_SEARCH_TERMS",
    "DEFAULT_SPLIT_FRACTIONS",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_DROPOUT",
    "DEFAULT_INPUT_FEATURES",
    # Loaders
    "TDCLoader",
    "ChEMBLAuroraLoader",
    "filter_aurora_data",
    "convert_aurora_to_standard",
    "infer_task_type",
    # Preprocessing
    "canonicalize_smiles",
    "validate_smiles_column",
    "deduplicate_by_label_consistency",
    "standardize_dataframe",
    "DataPreprocessor",
    # Splitting
    "create_scaffold_split",
    "create_random_split",
    "split_by_task",
    # Fragmentation
    "brics_decompose",
    "get_fragment_membership_matrix",
    "get_edge_break_mask",
    "FragmentExtractor",
    # Graph building
    "one_hot_encode",
    "get_atom_features",
    "mol_to_graph",
    "GraphBuilder",
    "save_graphs",
    "load_graphs",
]
