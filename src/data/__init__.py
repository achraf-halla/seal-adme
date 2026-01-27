"""
SEAL-ADME Data Module.

This module provides comprehensive data processing utilities for ADME property prediction:

1. **Data Loading** (loaders.py)
   - TDC dataset loading
   - ChEMBL Aurora kinase data fetching
   - Generated molecule loading

2. **Preprocessing** (preprocessing.py)
   - SMILES canonicalization and validation
   - Label-consistency deduplication
   - Train/valid/test splitting

3. **Fragmentation** (fragmentation.py)
   - BRICS molecular decomposition
   - Fragment SMILES extraction
   - Cluster assignment matrices

4. **Graph Featurization** (graph_featurizer.py)
   - Atom feature computation
   - PyTorch Geometric graph creation
   - Fragment-aware graph construction

Example Usage
-------------
>>> from src.data import (
...     load_pretrain_data,
...     load_finetune_data,
...     preprocess_dataset,
...     GraphFeaturizer,
... )
>>>
>>> # Load and preprocess data
>>> df = load_finetune_data()
>>> df_clean, stats = preprocess_dataset(df)
>>>
>>> # Create graphs
>>> featurizer = GraphFeaturizer(store_fragments=True)
>>> graphs = featurizer(df_clean, {"mean": 0.0, "std": 1.0})
"""

from .constants import (
    # Column names
    STANDARD_COLUMNS,
    REQUIRED_COLUMNS,
    # Task definitions
    PRETRAIN_TASKS,
    FINETUNE_TASKS,
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    # Atom features
    DEFAULT_ATOM_TYPE_SET,
    DEFAULT_HYBRIDIZATION_SET,
    PAULING_ELECTRONEGATIVITY,
    DEFAULT_ELECTRONEGATIVITY,
)

from .loaders import (
    # TDC loading
    load_tdc_task,
    load_tdc_tasks,
    load_pretrain_data,
    load_finetune_data,
    # ChEMBL loading
    find_aurora_target_ids,
    fetch_aurora_activities,
    load_aurora_data,
    # Generated molecules
    load_generated_molecules,
    # Utilities
    infer_task_type,
    read_csv_safe,
    save_manifest,
)

from .preprocessing import (
    # SMILES processing
    canonicalize_smiles,
    validate_smiles_column,
    # Deduplication
    deduplicate_by_label_consistency,
    check_drugid_smiles_mapping,
    # Summary
    summarize_dataset,
    # Full pipeline
    preprocess_dataset,
    # Splitting
    create_scaffold_split,
    create_random_split,
)

from .fragmentation import (
    # Fragment extraction
    FragmentExtractor,
    # BRICS decomposition
    brics_decomp,
    brics_decomp_extra,
    extract_fragment_metadata,
    # Cluster matrices
    create_cluster_assignment_matrix,
    create_atom_wise_assignment_matrix,
    # Edge masking
    mask_broken_edges,
    identify_inter_fragment_edges,
)

from .graph_featurizer import (
    # Featurization
    AtomFeaturizer,
    GraphFeaturizer,
    compute_gasteiger_charges,
    get_edge_index,
    featurize_dataset,
    one_hot_encoding,
)

__all__ = [
    # Constants
    "STANDARD_COLUMNS",
    "REQUIRED_COLUMNS",
    "PRETRAIN_TASKS",
    "FINETUNE_TASKS",
    "CLASSIFICATION_TASKS",
    "REGRESSION_TASKS",
    "DEFAULT_ATOM_TYPE_SET",
    "DEFAULT_HYBRIDIZATION_SET",
    "PAULING_ELECTRONEGATIVITY",
    "DEFAULT_ELECTRONEGATIVITY",
    # Loaders
    "load_tdc_task",
    "load_tdc_tasks",
    "load_pretrain_data",
    "load_finetune_data",
    "find_aurora_target_ids",
    "fetch_aurora_activities",
    "load_aurora_data",
    "load_generated_molecules",
    "infer_task_type",
    "read_csv_safe",
    "save_manifest",
    # Preprocessing
    "canonicalize_smiles",
    "validate_smiles_column",
    "deduplicate_by_label_consistency",
    "check_drugid_smiles_mapping",
    "summarize_dataset",
    "preprocess_dataset",
    "create_scaffold_split",
    "create_random_split",
    # Fragmentation
    "FragmentExtractor",
    "brics_decomp",
    "brics_decomp_extra",
    "extract_fragment_metadata",
    "create_cluster_assignment_matrix",
    "create_atom_wise_assignment_matrix",
    "mask_broken_edges",
    "identify_inter_fragment_edges",
    # Graph featurization
    "AtomFeaturizer",
    "GraphFeaturizer",
    "compute_gasteiger_charges",
    "get_edge_index",
    "featurize_dataset",
    "one_hot_encoding",
]
