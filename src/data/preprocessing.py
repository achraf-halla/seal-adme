"""
Preprocessing utilities for SEAL-ADME.

This module provides functions for:
- SMILES canonicalization and validation
- Label-consistency deduplication
- Data quality checks
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .constants import REQUIRED_COLUMNS, SMILES_ENCODINGS

logger = logging.getLogger(__name__)


# =============================================================================
# SMILES canonicalization
# =============================================================================

def canonicalize_smiles(
    smiles: str,
    isomeric: bool = True
) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        isomeric: If True, preserve stereochemistry
        
    Returns:
        Canonical SMILES or None if invalid
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit not installed. Run: pip install rdkit")
    
    if pd.isna(smiles):
        return None
    
    smiles_str = str(smiles).strip()
    if smiles_str == "":
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def validate_smiles_column(
    df: pd.DataFrame,
    smiles_col: str = "original_smiles",
    output_col: str = "canonical_smiles",
    isomeric: bool = True,
    drop_invalid: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate and canonicalize SMILES in a DataFrame.
    
    Args:
        df: Input DataFrame
        smiles_col: Name of column containing SMILES
        output_col: Name of column for canonical SMILES
        isomeric: If True, preserve stereochemistry
        drop_invalid: If True, remove rows with invalid SMILES
        
    Returns:
        Tuple of (processed DataFrame, statistics dict)
    """
    df = df.copy()
    n_input = len(df)
    
    # Canonicalize
    df[output_col] = df[smiles_col].apply(
        lambda x: canonicalize_smiles(x, isomeric=isomeric)
    )
    
    n_valid = df[output_col].notna().sum()
    n_invalid = n_input - n_valid
    
    stats = {
        "n_input": n_input,
        "n_valid": int(n_valid),
        "n_invalid": int(n_invalid),
    }
    
    if drop_invalid:
        df = df[df[output_col].notna()].copy()
        stats["n_output"] = len(df)
    else:
        stats["n_output"] = n_input
    
    logger.info(
        f"SMILES validation: {n_valid}/{n_input} valid "
        f"({n_invalid} invalid/unparsable)"
    )
    
    return df, stats


# =============================================================================
# Label-consistency deduplication
# =============================================================================

def normalize_label(y) -> str:
    """Normalize a label value for comparison."""
    if pd.isna(y):
        return "<NA>"
    return str(y).strip()


def deduplicate_by_label_consistency(
    df: pd.DataFrame,
    group_cols: List[str] = ["canonical_smiles", "task_name"],
    label_col: str = "Y",
    keep: str = "first",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Deduplicate by keeping only groups with consistent labels.
    
    For each group defined by (canonical_smiles, task_name):
    - If all labels are the same: keep first occurrence
    - If labels conflict: drop entire group
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        label_col: Column containing labels
        keep: Which row to keep ("first" or "last")
        
    Returns:
        Tuple of (deduplicated DataFrame, statistics dict)
    """
    df = df.copy()
    n_input = len(df)
    
    grouped = df.groupby(group_cols, sort=False)
    
    keep_indices = []
    n_consistent = 0
    n_conflicting = 0
    dropped_rows = 0
    
    for name, group in grouped:
        # Normalize labels for comparison
        labels_normalized = set(group[label_col].map(normalize_label).unique())
        
        if len(labels_normalized) == 1:
            # All labels consistent - keep first/last occurrence
            if keep == "first":
                keep_indices.append(group.index[0])
            else:
                keep_indices.append(group.index[-1])
            n_consistent += 1
        else:
            # Conflicting labels - drop entire group
            n_conflicting += 1
            dropped_rows += len(group)
    
    df_dedup = df.loc[keep_indices].copy()
    
    stats = {
        "n_input": n_input,
        "n_groups_consistent": n_consistent,
        "n_groups_conflicting": n_conflicting,
        "n_rows_dropped": dropped_rows,
        "n_output": len(df_dedup),
    }
    
    logger.info(
        f"Deduplication: kept {n_consistent} groups, "
        f"dropped {n_conflicting} conflicting groups ({dropped_rows} rows)"
    )
    
    return df_dedup, stats


# =============================================================================
# Drug_ID consistency checks
# =============================================================================

def check_drugid_smiles_mapping(
    df: pd.DataFrame,
    drugid_col: str = "Drug_ID",
    smiles_col: str = "canonical_smiles",
    task_col: str = "task_name",
) -> Dict[str, List[Dict]]:
    """
    Check if Drug_IDs consistently map to SMILES within each task.
    
    Args:
        df: Input DataFrame
        drugid_col: Column containing Drug IDs
        smiles_col: Column containing canonical SMILES
        task_col: Column containing task names
        
    Returns:
        Dictionary mapping task names to lists of conflicts
    """
    conflicts_by_task = {}
    
    if drugid_col not in df.columns or smiles_col not in df.columns:
        logger.warning(f"Required columns missing for mapping check")
        return conflicts_by_task
    
    # Convert Drug_ID to string for consistent comparison
    df = df.copy()
    df["_drug_id_str"] = df[drugid_col].map(
        lambda x: "" if pd.isna(x) else str(x).strip()
    )
    
    tasks = df[task_col].unique() if task_col in df.columns else ["all"]
    
    for task in tasks:
        if task_col in df.columns:
            sub = df[df[task_col] == task]
        else:
            sub = df
        
        # Count distinct SMILES per Drug_ID
        per_drug = sub.groupby("_drug_id_str")[smiles_col].nunique()
        multi_mapping = per_drug[per_drug > 1]
        
        if not multi_mapping.empty:
            conflicts = []
            for drug_id, count in multi_mapping.items():
                examples = sub[sub["_drug_id_str"] == drug_id][smiles_col].unique().tolist()
                conflicts.append({
                    "Drug_ID": drug_id,
                    "n_smiles": int(count),
                    "examples": examples[:5],  # Limit examples
                })
            conflicts_by_task[task] = conflicts
            logger.warning(
                f"Task '{task}': {len(conflicts)} Drug_IDs map to multiple SMILES"
            )
    
    return conflicts_by_task


# =============================================================================
# Data quality summary
# =============================================================================

def summarize_dataset(
    df: pd.DataFrame,
    task_col: str = "task_name",
    label_col: str = "Y",
    smiles_col: str = "canonical_smiles",
) -> Dict:
    """
    Generate a summary of dataset quality and statistics.
    
    Args:
        df: Input DataFrame
        task_col: Column containing task names
        label_col: Column containing labels
        smiles_col: Column containing SMILES
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_rows": len(df),
        "n_unique_smiles": df[smiles_col].nunique() if smiles_col in df.columns else None,
        "nan_counts": df.isna().sum().to_dict(),
        "tasks": {},
    }
    
    if task_col not in df.columns:
        return summary
    
    for task in df[task_col].unique():
        sub = df[df[task_col] == task]
        task_type = sub.get("task", pd.Series(["unknown"])).iloc[0]
        
        task_summary = {
            "n_samples": len(sub),
            "n_unique_molecules": sub["Drug_ID"].nunique() if "Drug_ID" in sub.columns else None,
            "task_type": task_type,
        }
        
        # Add label statistics
        if label_col in sub.columns:
            if task_type == "classification":
                task_summary["class_distribution"] = sub[label_col].value_counts().to_dict()
            else:
                y_numeric = pd.to_numeric(sub[label_col], errors="coerce")
                task_summary["label_stats"] = {
                    "mean": float(y_numeric.mean()),
                    "std": float(y_numeric.std()),
                    "min": float(y_numeric.min()),
                    "max": float(y_numeric.max()),
                }
        
        summary["tasks"][task] = task_summary
    
    return summary


# =============================================================================
# Full preprocessing pipeline
# =============================================================================

def preprocess_dataset(
    df: pd.DataFrame,
    smiles_col: str = "original_smiles",
    isomeric: bool = True,
    deduplicate: bool = True,
    drop_invalid: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full preprocessing pipeline on a dataset.
    
    Steps:
    1. Validate and canonicalize SMILES
    2. Deduplicate by label consistency
    3. Check Drug_ID -> SMILES mapping
    
    Args:
        df: Input DataFrame
        smiles_col: Name of SMILES column
        isomeric: Preserve stereochemistry
        deduplicate: Apply label-consistency deduplication
        drop_invalid: Drop rows with invalid SMILES
        
    Returns:
        Tuple of (processed DataFrame, statistics dict)
    """
    all_stats = {"steps": {}}
    
    # Step 1: Validate SMILES
    df, smiles_stats = validate_smiles_column(
        df, 
        smiles_col=smiles_col,
        isomeric=isomeric,
        drop_invalid=drop_invalid,
    )
    all_stats["steps"]["smiles_validation"] = smiles_stats
    
    # Step 2: Deduplicate
    if deduplicate and "canonical_smiles" in df.columns:
        df, dedup_stats = deduplicate_by_label_consistency(df)
        all_stats["steps"]["deduplication"] = dedup_stats
    
    # Step 3: Check mapping consistency
    if "canonical_smiles" in df.columns:
        conflicts = check_drugid_smiles_mapping(df)
        all_stats["drugid_conflicts"] = {
            k: len(v) for k, v in conflicts.items()
        }
    
    # Final summary
    all_stats["final"] = summarize_dataset(df)
    
    return df, all_stats


# =============================================================================
# Train/Valid/Test splitting
# =============================================================================

def create_scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data using Murcko scaffold splitting.
    
    Args:
        df: Input DataFrame
        smiles_col: Column containing SMILES
        train_ratio: Fraction for training
        valid_ratio: Fraction for validation
        seed: Random seed
        
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        raise ImportError("RDKit not installed")
    
    from collections import defaultdict
    import random
    
    random.seed(seed)
    
    # Compute scaffolds
    scaffold_to_indices = defaultdict(list)
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles):
            scaffold = "NONE"
        else:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                else:
                    scaffold = "INVALID"
            except Exception:
                scaffold = "ERROR"
        scaffold_to_indices[scaffold].append(idx)
    
    # Shuffle scaffolds
    scaffolds = list(scaffold_to_indices.keys())
    random.shuffle(scaffolds)
    
    # Assign scaffolds to splits
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    train_indices = []
    valid_indices = []
    test_indices = []
    
    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]
        if len(train_indices) < n_train:
            train_indices.extend(indices)
        elif len(valid_indices) < n_valid:
            valid_indices.extend(indices)
        else:
            test_indices.extend(indices)
    
    train_df = df.loc[train_indices].reset_index(drop=True)
    valid_df = df.loc[valid_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    
    logger.info(
        f"Scaffold split: train={len(train_df)}, "
        f"valid={len(valid_df)}, test={len(test_df)}"
    )
    
    return train_df, valid_df, test_df


def create_random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data randomly, optionally with stratification.
    
    Args:
        df: Input DataFrame
        train_ratio: Fraction for training
        valid_ratio: Fraction for validation
        seed: Random seed
        stratify_col: Column to stratify by (for classification)
        
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("scikit-learn not installed")
    
    test_ratio = 1.0 - train_ratio - valid_ratio
    
    stratify = df[stratify_col] if stratify_col else None
    
    # First split: train+valid vs test
    train_valid_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )
    
    # Second split: train vs valid
    relative_valid = valid_ratio / (train_ratio + valid_ratio)
    stratify_tv = train_valid_df[stratify_col] if stratify_col else None
    
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=relative_valid,
        random_state=seed,
        stratify=stratify_tv,
    )
    
    logger.info(
        f"Random split: train={len(train_df)}, "
        f"valid={len(valid_df)}, test={len(test_df)}"
    )
    
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
