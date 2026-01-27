"""
Data loaders for SEAL-ADME.

This module provides functions to load ADME datasets from:
- Therapeutics Data Commons (TDC)
- ChEMBL (Aurora kinase bioactivity data)
- Generated molecules
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .constants import (
    PRETRAIN_TASKS,
    FINETUNE_TASKS,
    MIN_CLASSIFICATION_THRESHOLD,
    AURORA_TARGET_SYNONYMS,
    AURORA_ACTIVITY_TYPES,
    SMILES_ENCODINGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Task type inference
# =============================================================================

def infer_task_type(series: pd.Series, threshold: int = MIN_CLASSIFICATION_THRESHOLD) -> str:
    """
    Infer whether a task is classification or regression based on label distribution.
    
    Args:
        series: Pandas Series of target values
        threshold: Maximum unique values to be considered classification
        
    Returns:
        "classification" or "regression"
    """
    s = series.dropna()
    if s.empty:
        return "classification"
    
    # Try to convert to numeric
    s_num = pd.to_numeric(s, errors="coerce")
    num_numeric = s_num.notna().sum()
    
    # If less than 80% are numeric, treat as classification
    if num_numeric < max(1, int(0.8 * len(s))):
        return "classification"
    
    nunique = int(s_num.nunique())
    return "classification" if nunique <= threshold else "regression"


# =============================================================================
# TDC data loading
# =============================================================================

def load_tdc_task(task_name: str, source_tag: str = "tdc") -> pd.DataFrame:
    """
    Load a single ADME task from TDC.
    
    Args:
        task_name: Name of the TDC ADME task
        source_tag: Tag to identify data source (e.g., "pretrain", "finetune")
        
    Returns:
        DataFrame with standardized columns
    """
    try:
        from tdc.single_pred import ADME
    except ImportError:
        raise ImportError("TDC not installed. Run: pip install PyTDC")
    
    adme = ADME(name=task_name)
    df = adme.get_data()
    
    # Standardize column names
    df = df.rename(columns={"Drug": "original_smiles"})
    df["task_name"] = task_name
    df["source"] = source_tag
    df["task"] = infer_task_type(df["Y"])
    
    # Select and order columns
    columns = ["Drug_ID", "original_smiles", "Y", "task_name", "source", "task"]
    df = df[columns]
    
    logger.info(f"Loaded {task_name}: {len(df)} samples ({df['task'].iloc[0]})")
    return df


def load_tdc_tasks(
    tasks: List[str],
    source_tag: str = "tdc"
) -> pd.DataFrame:
    """
    Load multiple ADME tasks from TDC.
    
    Args:
        tasks: List of TDC task names
        source_tag: Tag to identify data source
        
    Returns:
        Combined DataFrame with all tasks
    """
    dfs = []
    for task in tasks:
        try:
            df = load_tdc_task(task, source_tag)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {task}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=[
            "Drug_ID", "original_smiles", "Y", "task_name", "source", "task"
        ])
    
    return pd.concat(dfs, ignore_index=True)


def load_pretrain_data() -> pd.DataFrame:
    """Load all pretraining tasks from TDC."""
    return load_tdc_tasks(PRETRAIN_TASKS, source_tag="pretrain")


def load_finetune_data() -> pd.DataFrame:
    """Load all finetuning tasks from TDC."""
    return load_tdc_tasks(FINETUNE_TASKS, source_tag="finetune")


# =============================================================================
# ChEMBL data loading (Aurora kinase)
# =============================================================================

def find_aurora_target_ids() -> List[str]:
    """
    Find ChEMBL target IDs for Aurora kinases.
    
    Returns:
        List of ChEMBL target IDs
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "ChEMBL client not installed. Run: pip install chembl-webresource-client"
        )
    
    target_client = new_client.target
    
    # Search by preferred name
    targets = list(target_client.filter(pref_name__icontains="Aurora"))
    
    # Search by synonyms
    for synonym in AURORA_TARGET_SYNONYMS:
        try:
            targets.extend(list(target_client.filter(target_synonym__icontains=synonym)))
        except Exception:
            continue
    
    # Extract unique target IDs
    target_ids = {t.get("target_chembl_id") for t in targets if t.get("target_chembl_id")}
    return list(target_ids)


def fetch_aurora_activities(target_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch Aurora kinase bioactivity data from ChEMBL.
    
    Args:
        target_ids: Optional list of target IDs. If None, will be auto-discovered.
        
    Returns:
        DataFrame with activity data
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "ChEMBL client not installed. Run: pip install chembl-webresource-client"
        )
    
    if target_ids is None:
        target_ids = find_aurora_target_ids()
    
    if not target_ids:
        logger.warning("No Aurora kinase targets found")
        return pd.DataFrame()
    
    logger.info(f"Fetching activities for {len(target_ids)} Aurora kinase targets")
    
    activity_client = new_client.activity
    assay_client = new_client.assay
    
    # Cache for assay metadata
    assay_cache: Dict[str, Dict] = {}
    
    rows = []
    query = activity_client.filter(
        target_chembl_id__in=target_ids
    ).filter(
        standard_type__in=AURORA_ACTIVITY_TYPES
    )
    
    for rec in query:
        smiles = rec.get("canonical_smiles") or rec.get("smiles")
        sval = rec.get("standard_value")
        sval_num = pd.to_numeric(sval, errors="coerce")
        
        if not smiles or pd.isna(sval_num):
            continue
        
        # Fetch assay metadata
        assay_id = rec.get("assay_chembl_id")
        if assay_id and assay_id not in assay_cache:
            try:
                assay = assay_client.get(assay_id)
                assay_cache[assay_id] = {
                    "assay_chembl_id": assay_id,
                    "assay_description": assay.get("description", ""),
                    "assay_type": assay.get("assay_type", ""),
                    "assay_organism": assay.get("assay_organism", ""),
                }
            except Exception:
                assay_cache[assay_id] = {"assay_chembl_id": assay_id}
        
        assay_meta = assay_cache.get(assay_id, {})
        
        rows.append({
            "molecule_chembl_id": rec.get("molecule_chembl_id"),
            "original_smiles": smiles,
            "standard_value": float(sval_num),
            "standard_units": rec.get("standard_units"),
            "standard_relation": rec.get("standard_relation"),
            "target_chembl_id": rec.get("target_chembl_id"),
            "standard_type": rec.get("standard_type"),
            "pchembl_value": rec.get("pchembl_value"),
            **assay_meta,
        })
    
    df = pd.DataFrame(rows)
    logger.info(f"Fetched {len(df)} Aurora kinase activity records")
    return df


def load_aurora_data(
    output_path: Optional[Path] = None,
    force_fetch: bool = False
) -> pd.DataFrame:
    """
    Load Aurora kinase data, fetching from ChEMBL if necessary.
    
    Args:
        output_path: Optional path to cache the data
        force_fetch: If True, always fetch fresh data from ChEMBL
        
    Returns:
        Standardized DataFrame with Aurora kinase data
    """
    # Check for cached data
    if output_path and output_path.exists() and not force_fetch:
        logger.info(f"Loading cached Aurora data from {output_path}")
        df = pd.read_csv(output_path)
    else:
        df = fetch_aurora_activities()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved Aurora data to {output_path}")
    
    if df.empty:
        return df
    
    # Standardize to match TDC format
    df_std = df.copy()
    df_std["Drug_ID"] = df_std["molecule_chembl_id"]
    df_std["Y"] = df_std["pchembl_value"]  # Use pChEMBL value as target
    df_std["task_name"] = "aurora_potency"
    df_std["source"] = "chembl"
    df_std["task"] = "regression"
    
    return df_std


# =============================================================================
# Generated molecules loading
# =============================================================================

def load_generated_molecules(
    path_or_url: Union[str, Path],
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    """
    Load generated molecules from a CSV file or URL.
    
    Args:
        path_or_url: Path to CSV file or URL
        smiles_col: Name of the SMILES column
        
    Returns:
        Standardized DataFrame
    """
    df = pd.read_csv(str(path_or_url))
    
    # Standardize column names
    if smiles_col in df.columns and smiles_col != "original_smiles":
        df = df.rename(columns={smiles_col: "original_smiles"})
    
    # Remove Name column if present
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])
    
    # Add required columns
    df["Y"] = 0  # Placeholder for unlabeled data
    df["task"] = "generated"
    df["task_name"] = "generated"
    df["source"] = "generated"
    
    # Generate Drug_IDs if not present
    if "Drug_ID" not in df.columns:
        df.insert(0, "Drug_ID", [f"GEN_{i+1}" for i in range(len(df))])
    
    # Select columns
    columns = ["Drug_ID", "original_smiles", "Y", "task_name", "source", "task"]
    df = df[[c for c in columns if c in df.columns]]
    
    logger.info(f"Loaded {len(df)} generated molecules")
    return df


# =============================================================================
# Utility functions
# =============================================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Read CSV file with multiple encoding fallbacks.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame
    """
    for encoding in SMILES_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    
    # Final fallback with Python engine
    return pd.read_csv(path, engine="python")


def save_manifest(
    manifest: Dict,
    output_path: Path,
    step_name: str = "data_loading"
) -> None:
    """
    Save a manifest file documenting the data loading step.
    
    Args:
        manifest: Dictionary with manifest data
        output_path: Path to save manifest
        step_name: Name of the processing step
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["step"] = step_name
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {output_path}")
