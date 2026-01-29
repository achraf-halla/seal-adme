"""
Data loaders for TDC ADME datasets and ChEMBL Aurora kinase data.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from .constants import PRETRAIN_TASKS, FINETUNE_TASKS, AURORA_SEARCH_TERMS, STANDARD_COLUMNS

logger = logging.getLogger(__name__)


def infer_task_type(series: pd.Series) -> str:
    """
    Infer whether a task is classification or regression based on label distribution.
    
    Args:
        series: Pandas series of labels
        
    Returns:
        'classification' or 'regression'
    """
    s = series.dropna()
    if s.empty:
        return "classification"
    
    s_num = pd.to_numeric(s, errors="coerce")
    num_numeric = s_num.notna().sum()
    
    if num_numeric < max(1, int(0.8 * len(s))):
        return "classification"
    
    nunique = int(s_num.nunique())
    return "classification" if nunique <= 10 else "regression"


class TDCLoader:
    """Load ADME datasets from Therapeutics Data Commons (TDC)."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize TDC loader.
        
        Args:
            output_dir: Directory to save raw data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_tasks(
        self,
        tasks: List[str],
        source_tag: str
    ) -> pd.DataFrame:
        """
        Load multiple TDC tasks and combine into single DataFrame.
        
        Args:
            tasks: List of TDC task names
            source_tag: Tag to identify data source ('pretrain' or 'finetune')
            
        Returns:
            Combined DataFrame with standardized columns
        """
        try:
            from tdc.single_pred import ADME
        except ImportError:
            raise ImportError("pytdc is required. Install with: pip install pytdc")
        
        dfs = []
        for task in tasks:
            logger.info(f"Loading TDC task: {task}")
            try:
                adme = ADME(name=task)
                df = adme.get_data()
                df = df.rename(columns={"Drug": "original_smiles"})
                df["task_name"] = task
                df["source"] = source_tag
                df["task"] = infer_task_type(df["Y"])
                df = df[STANDARD_COLUMNS]
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load task {task}: {e}")
                continue
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    def load_pretrain(self) -> pd.DataFrame:
        """Load all pretraining tasks."""
        return self.load_tasks(PRETRAIN_TASKS, "pretrain")
    
    def load_finetune(self) -> pd.DataFrame:
        """Load all finetuning tasks."""
        return self.load_tasks(FINETUNE_TASKS, "finetune")
    
    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path


class ChEMBLAuroraLoader:
    """Load Aurora kinase activity data from ChEMBL."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize ChEMBL loader.
        
        Args:
            output_dir: Directory to save raw data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._assay_cache: Dict[str, Dict] = {}
    
    def _get_client(self):
        """Get ChEMBL web resource client."""
        try:
            from chembl_webresource_client.new_client import new_client
            return new_client
        except ImportError:
            raise ImportError(
                "chembl_webresource_client is required. "
                "Install with: pip install chembl-webresource-client"
            )
    
    def find_aurora_targets(self) -> List[str]:
        """
        Find ChEMBL target IDs for Aurora kinases.
        
        Returns:
            List of target ChEMBL IDs
        """
        client = self._get_client()
        target_client = client.target
        
        # Search by preferred name
        pref = list(target_client.filter(pref_name__icontains="Aurora"))
        
        # Search by synonyms
        syn = []
        for term in AURORA_SEARCH_TERMS:
            try:
                syn.extend(list(target_client.filter(target_synonym__icontains=term)))
            except Exception:
                continue
        
        ids = {t.get("target_chembl_id") for t in (pref + syn) if t.get("target_chembl_id")}
        logger.info(f"Found {len(ids)} Aurora kinase targets")
        return list(ids)
    
    def _fetch_assay_metadata(self, assay_id: Optional[str]) -> Dict[str, Any]:
        """Fetch and cache assay metadata."""
        empty_meta = {
            "assay_chembl_id": None,
            "assay_description": "",
            "assay_type": "",
            "assay_organism": "",
            "assay_parameters": ""
        }
        
        if not assay_id:
            return empty_meta
        
        if assay_id in self._assay_cache:
            return self._assay_cache[assay_id]
        
        client = self._get_client()
        try:
            a = client.assay.get(assay_id)
            meta = {
                "assay_chembl_id": assay_id,
                "assay_description": a.get("description") or a.get("assay_description") or "",
                "assay_type": a.get("assay_type") or "",
                "assay_organism": a.get("assay_organism") or "",
                "assay_parameters": a.get("assay_parameters") or a.get("assay_param") or ""
            }
        except Exception:
            meta = {**empty_meta, "assay_chembl_id": assay_id}
        
        self._assay_cache[assay_id] = meta
        return meta
    
    def fetch_activities(
        self,
        target_ids: List[str],
        standard_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch activity data for specified targets.
        
        Args:
            target_ids: List of ChEMBL target IDs
            standard_types: Activity types to include (default: IC50, Ki)
            
        Returns:
            DataFrame with activity data
        """
        if standard_types is None:
            standard_types = ["IC50", "Ki"]
        
        client = self._get_client()
        activity_client = client.activity
        
        if not target_ids:
            logger.warning("No target IDs provided")
            return pd.DataFrame()
        
        query = activity_client.filter(
            target_chembl_id__in=target_ids
        ).filter(
            standard_type__in=standard_types
        )
        
        rows = []
        for rec in query:
            smi = rec.get("canonical_smiles") or rec.get("smiles")
            sval = rec.get("standard_value")
            sval_num = pd.to_numeric(sval, errors="coerce")
            
            if not smi or pd.isna(sval_num):
                continue
            
            assay_id = rec.get("assay_chembl_id")
            assay_meta = self._fetch_assay_metadata(assay_id)
            
            rows.append({
                "molecule_chembl_id": rec.get("molecule_chembl_id"),
                "canonical_smiles": smi,
                "standard_value": float(sval_num),
                "standard_units": rec.get("standard_units") or "",
                "standard_relation": rec.get("standard_relation") or "",
                "target_chembl_id": rec.get("target_chembl_id"),
                "standard_type": rec.get("standard_type"),
                "pchembl_value": rec.get("pchembl_value"),
                **assay_meta
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} activity records")
        return df
    
    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path


def filter_aurora_data(
    df: pd.DataFrame,
    min_target_count: int = 200,
    organism: str = "Homo sapiens",
    assay_type: str = "B",
    require_pchembl: bool = True
) -> pd.DataFrame:
    """
    Apply standard filters to Aurora kinase data.
    
    Args:
        df: Raw Aurora kinase DataFrame
        min_target_count: Minimum occurrences per target to keep
        organism: Filter to specific organism
        assay_type: Filter to assay type (B=binding)
        require_pchembl: Whether to require pChEMBL values
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Starting with {len(df)} rows")
    
    # Keep only exact measurements
    if "standard_relation" in df.columns:
        df = df[df["standard_relation"] == "="].copy()
        logger.info(f"After relation filter: {len(df)} rows")
    
    # Keep targets with sufficient data
    if "target_chembl_id" in df.columns:
        counts = df["target_chembl_id"].value_counts(dropna=True)
        keep_ids = counts[counts > min_target_count].index.tolist()
        df = df[df["target_chembl_id"].isin(keep_ids)].copy()
        logger.info(f"After target count filter (>{min_target_count}): {len(df)} rows")
    
    # Filter by organism
    if "assay_organism" in df.columns and organism:
        df = df[df["assay_organism"] == organism].copy()
        logger.info(f"After organism filter ({organism}): {len(df)} rows")
    
    # Filter by assay type
    if "assay_type" in df.columns and assay_type:
        df = df[df["assay_type"] == assay_type].copy()
        logger.info(f"After assay type filter ({assay_type}): {len(df)} rows")
    
    # Require pChEMBL values
    if require_pchembl and "pchembl_value" in df.columns:
        df = df[~df["pchembl_value"].isna()].copy()
        logger.info(f"After pChEMBL filter: {len(df)} rows")
    
    return df
