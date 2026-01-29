"""
Data processing pipeline for SEAL-ADME.

This module provides utilities for:
- Loading data from TDC and ChEMBL
- SMILES validation and canonicalization
- Molecular featurization (descriptors and fingerprints)
- BRICS-based fragmentation
- PyTorch Geometric graph creation
"""

from .constants import (
    PRETRAIN_TASKS,
    FINETUNE_TASKS,
    DEFAULT_ATOM_TYPE_SET,
    DEFAULT_HYBRIDIZATION_SET,
    FP_NBITS,
    FP_RADIUS,
)

from .loaders import (
    TDCLoader,
    ChEMBLAuroraLoader,
    filter_aurora_data,
    infer_task_type,
)

from .preprocessing import (
    canonicalize_smiles,
    validate_smiles_column,
    deduplicate_by_label_consistency,
    DataPreprocessor,
)

from .featurization import (
    MorganFingerprintCalculator,
    RDKitDescriptorCalculator,
    MolecularFeaturizer,
    impute_missing_descriptors,
)

from .fragmentation import (
    brics_decompose,
    FragmentExtractor,
)

from .graph_featurizer import (
    AtomFeaturizer,
    GraphFeaturizer,
    save_graphs,
    load_graphs,
)

__all__ = [
    # Constants
    "PRETRAIN_TASKS",
    "FINETUNE_TASKS", 
    "DEFAULT_ATOM_TYPE_SET",
    "DEFAULT_HYBRIDIZATION_SET",
    "FP_NBITS",
    "FP_RADIUS",
    # Loaders
    "TDCLoader",
    "ChEMBLAuroraLoader",
    "filter_aurora_data",
    "infer_task_type",
    # Preprocessing
    "canonicalize_smiles",
    "validate_smiles_column",
    "deduplicate_by_label_consistency",
    "DataPreprocessor",
    # Featurization
    "MorganFingerprintCalculator",
    "RDKitDescriptorCalculator",
    "MolecularFeaturizer",
    "impute_missing_descriptors",
    # Fragmentation
    "brics_decompose",
    "FragmentExtractor",
    # Graph featurization
    "AtomFeaturizer",
    "GraphFeaturizer",
    "save_graphs",
    "load_graphs",
]
