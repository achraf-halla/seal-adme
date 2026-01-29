"""
Data splitting utilities for SEAL-ADME.

Implements scaffold-based splitting to ensure structurally distinct
train/validation/test sets.
"""

import logging
from collections import defaultdict
from random import Random
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def print_sys(msg: str):
    """Print system message."""
    logger.warning(msg)


def create_scaffold_split(
    df: pd.DataFrame,
    seed: int,
    frac: List[float],
    entity: str
) -> Dict[str, pd.DataFrame]:
    """
    Create scaffold-based data split.
    
    Generates molecular scaffolds for each molecule and splits based on
    scaffold membership to ensure structurally distinct sets.
    
    Reference:
        https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py

    Args:
        df: Dataset DataFrame
        seed: Random seed for reproducibility
        frac: List of train/valid/test fractions (should sum to 1.0)
        entity: Column name containing SMILES strings

    Returns:
        Dictionary with 'train', 'valid', 'test' DataFrames
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        raise ImportError(
            "RDKit is required. Install with: conda install -c conda-forge rdkit"
        )

    random = Random(seed)

    smiles_list = df[entity].values
    scaffolds = defaultdict(set)

    error_smiles = 0
    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Computing scaffolds"):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles),
                includeChirality=False
            )
            scaffolds[scaffold].add(i)
        except Exception:
            print_sys(f"{smiles} returns RDKit error and is thus omitted...")
            error_smiles += 1

    train, val, test = [], [], []
    n_valid = len(df) - error_smiles
    train_size = int(n_valid * frac[0])
    val_size = int(n_valid * frac[1])
    test_size = n_valid - train_size - val_size

    # Sort scaffolds by size for better distribution
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    # Assign scaffolds to splits
    if frac[2] == 0:
        # No test set
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
            else:
                val += index_set
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
            elif len(val) + len(index_set) <= val_size:
                val += index_set
            else:
                test += index_set

    logger.info(
        f"Scaffold split: train={len(train)}, valid={len(val)}, test={len(test)}, "
        f"errors={error_smiles}"
    )

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[val].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }


def create_random_split(
    df: pd.DataFrame,
    seed: int,
    frac: List[float]
) -> Dict[str, pd.DataFrame]:
    """
    Create random data split.

    Args:
        df: Dataset DataFrame
        seed: Random seed
        frac: List of train/valid/test fractions

    Returns:
        Dictionary with 'train', 'valid', 'test' DataFrames
    """
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    n = len(df_shuffled)
    train_end = int(n * frac[0])
    val_end = train_end + int(n * frac[1])

    return {
        "train": df_shuffled.iloc[:train_end].reset_index(drop=True),
        "valid": df_shuffled.iloc[train_end:val_end].reset_index(drop=True),
        "test": df_shuffled.iloc[val_end:].reset_index(drop=True),
    }


def split_by_task(
    df: pd.DataFrame,
    task_col: str = "task_name",
    seed: int = 42,
    frac: List[float] = None,
    smiles_col: str = "canonical_smiles",
    method: str = "scaffold"
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split data by task using scaffold or random splitting.

    Args:
        df: Combined DataFrame with multiple tasks
        task_col: Column containing task names
        seed: Random seed
        frac: Train/valid/test fractions
        smiles_col: Column containing SMILES
        method: 'scaffold' or 'random'

    Returns:
        Nested dict: {task_name: {'train': df, 'valid': df, 'test': df}}
    """
    if frac is None:
        frac = [0.8, 0.1, 0.1]

    tasks = df[task_col].unique()
    task_splits = {}

    for task in tasks:
        task_df = df[df[task_col] == task].copy()
        
        if method == "scaffold":
            splits = create_scaffold_split(task_df, seed, frac, smiles_col)
        else:
            splits = create_random_split(task_df, seed, frac)
        
        task_splits[task] = splits
        logger.info(
            f"Task '{task}': train={len(splits['train'])}, "
            f"valid={len(splits['valid'])}, test={len(splits['test'])}"
        )

    return task_splits
