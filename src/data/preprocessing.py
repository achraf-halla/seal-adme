"""
SMILES validation, canonicalization, and deduplication utilities.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: Any, isomeric: bool = True) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        isomeric: Whether to preserve stereochemistry
        
    Returns:
        Canonical SMILES or None if invalid
    """
    from rdkit import Chem
    
    try:
        if pd.isna(smiles):
            return None
        sm = str(smiles).strip()
        if sm == "":
            return None
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def validate_smiles_column(
    df: pd.DataFrame,
    smiles_col: str = "original_smiles",
    output_col: str = "canonical_smiles"
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate and canonicalize SMILES in a DataFrame.
    
    Args:
        df: Input DataFrame
        smiles_col: Name of column containing SMILES
        output_col: Name of column for canonical SMILES
        
    Returns:
        Tuple of (DataFrame with canonical SMILES, validation stats)
    """
    df = df.copy()
    
    n_missing = df[smiles_col].isna().sum()
    n_blank = (df[smiles_col].astype(str).str.strip() == "").sum()
    
    df[output_col] = df[smiles_col].map(canonicalize_smiles)
    
    n_valid = df[output_col].notna().sum()
    n_invalid = len(df) - n_valid
    
    stats = {
        "total": len(df),
        "missing": int(n_missing),
        "blank": int(n_blank),
        "valid": int(n_valid),
        "invalid": int(n_invalid)
    }
    
    logger.info(f"SMILES validation: {n_valid} valid, {n_invalid} invalid out of {len(df)}")
    
    return df, stats


def deduplicate_by_label_consistency(
    df: pd.DataFrame,
    group_cols: List[str] = None,
    label_col: str = "Y"
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Deduplicate DataFrame by keeping only groups with consistent labels.
    
    For each unique combination of group_cols, if all labels are the same,
    keep one representative row. If labels conflict, drop the entire group.
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by (default: canonical_smiles, task_name)
        label_col: Column containing labels
        
    Returns:
        Tuple of (deduplicated DataFrame, deduplication stats)
    """
    if group_cols is None:
        group_cols = ["canonical_smiles", "task_name"]
    
    df = df.copy()
    
    def normalize_label(y):
        if pd.isna(y):
            return "<NA>"
        return str(y).strip()
    
    grouped = df.groupby(group_cols, sort=False)
    
    keep_indices = []
    keep_groups = 0
    drop_groups = 0
    dropped_rows = 0
    
    for name, group in grouped:
        labels_norm = set(group[label_col].map(normalize_label).unique())
        if len(labels_norm) == 1:
            keep_indices.append(group.index[0])
            keep_groups += 1
        else:
            drop_groups += 1
            dropped_rows += len(group)
    
    df_dedup = df.loc[keep_indices].copy()
    
    stats = {
        "groups_kept": keep_groups,
        "groups_dropped": drop_groups,
        "rows_before": len(df),
        "rows_after": len(df_dedup),
        "rows_dropped": dropped_rows
    }
    
    logger.info(
        f"Deduplication: kept {keep_groups} groups, "
        f"dropped {drop_groups} groups ({dropped_rows} rows)"
    )
    
    return df_dedup, stats


def standardize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "original_smiles",
    drop_invalid: bool = True,
    deduplicate: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply full standardization pipeline to a DataFrame.
    
    Pipeline:
    1. Validate and canonicalize SMILES
    2. Drop invalid SMILES (optional)
    3. Deduplicate by label consistency (optional)
    
    Args:
        df: Input DataFrame
        smiles_col: Column containing SMILES
        drop_invalid: Whether to drop rows with invalid SMILES
        deduplicate: Whether to deduplicate by label consistency
        
    Returns:
        Tuple of (standardized DataFrame, processing stats)
    """
    stats = {"input_rows": len(df)}
    
    # Validate and canonicalize
    df, val_stats = validate_smiles_column(df, smiles_col, "canonical_smiles")
    stats["validation"] = val_stats
    
    # Drop invalid
    if drop_invalid:
        n_before = len(df)
        df = df[df["canonical_smiles"].notna()].copy()
        stats["dropped_invalid"] = n_before - len(df)
    
    # Deduplicate
    if deduplicate:
        df, dedup_stats = deduplicate_by_label_consistency(df)
        stats["deduplication"] = dedup_stats
    
    stats["output_rows"] = len(df)
    
    return df, stats


class DataPreprocessor:
    """
    Pipeline for preprocessing molecular data.
    
    Handles validation, canonicalization, and deduplication.
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        smiles_col: str = "original_smiles"
    ):
        """
        Initialize preprocessor.
        
        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory to save preprocessed files
            smiles_col: Name of SMILES column
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.smiles_col = smiles_col
    
    def process_file(
        self,
        filename: str,
        drop_invalid: bool = True,
        deduplicate: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a single CSV file.
        
        Args:
            filename: Name of input file
            drop_invalid: Whether to drop invalid SMILES
            deduplicate: Whether to deduplicate by label consistency
            
        Returns:
            Tuple of (processed DataFrame, processing stats)
        """
        input_path = self.input_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        df = self._read_csv(input_path)
        logger.info(f"Processing {filename}: {len(df)} rows")
        
        df, stats = standardize_dataframe(
            df,
            smiles_col=self.smiles_col,
            drop_invalid=drop_invalid,
            deduplicate=deduplicate
        )
        
        return df, stats
    
    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read CSV with fallback encodings."""
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path, engine="python")
    
    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """Save processed DataFrame."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path
    
    def save_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """Save processed DataFrame as parquet."""
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path
