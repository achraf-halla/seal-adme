"""
Molecular featurization using RDKit descriptors and fingerprints.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import FP_NBITS, FP_RADIUS

logger = logging.getLogger(__name__)


def safe_compute(func, *args, default=None, **kwargs):
    """Safely compute a function, returning default on error."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default


class MorganFingerprintCalculator:
    """Calculate Morgan (circular) fingerprints."""
    
    def __init__(self, n_bits: int = FP_NBITS, radius: int = FP_RADIUS):
        """
        Initialize fingerprint calculator.
        
        Args:
            n_bits: Number of bits in fingerprint
            radius: Fingerprint radius
        """
        self.n_bits = n_bits
        self.radius = radius
    
    def compute(self, mol) -> Tuple[List[int], str]:
        """
        Compute Morgan fingerprint as bit vector and packed hex string.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Tuple of (bit list, hex-packed string)
        """
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        
        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )
        
        arr = np.zeros((self.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        bits = arr.tolist()
        
        # Pack to hex string
        bitstring = "".join("1" if x else "0" for x in bits)
        pad = (-len(bitstring)) % 8
        if pad:
            bitstring = ("0" * pad) + bitstring
        packed = int(bitstring, 2).to_bytes(len(bitstring) // 8, byteorder="big").hex()
        
        return bits, packed
    
    def compute_from_smiles(self, smiles: str) -> Optional[Tuple[List[int], str]]:
        """Compute fingerprint from SMILES string."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.compute(mol)


class RDKitDescriptorCalculator:
    """Calculate all RDKit molecular descriptors."""
    
    def __init__(self, missing_val: float = None, silent: bool = True):
        """
        Initialize descriptor calculator.
        
        Args:
            missing_val: Value to use for missing/failed descriptors
            silent: Whether to suppress RDKit warnings
        """
        self.missing_val = missing_val
        self.silent = silent
    
    def compute(self, mol) -> Dict[str, Any]:
        """
        Calculate all RDKit descriptors for a molecule.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary of descriptor name -> value
        """
        from rdkit.Chem import Descriptors
        
        out = {}
        
        # Try CalcMolDescriptors first (newer RDKit)
        try:
            if hasattr(Descriptors, 'CalcMolDescriptors'):
                vals = Descriptors.CalcMolDescriptors(
                    mol, missingVal=self.missing_val, silent=self.silent
                )
                if isinstance(vals, dict):
                    out.update(vals)
                    return out
                elif isinstance(vals, (list, tuple)):
                    if hasattr(Descriptors, 'descList'):
                        names = [n for n, _ in Descriptors.descList]
                        for n, v in zip(names, vals):
                            out[n] = self.missing_val if (
                                isinstance(v, float) and np.isnan(v)
                            ) else v
                        return out
        except Exception as e:
            if not self.silent:
                logger.warning(f"CalcMolDescriptors failed: {e}")
        
        # Fallback: iterate through descriptor list
        desc_list = getattr(Descriptors, 'descList', None) or getattr(
            Descriptors, '_descList', None
        )
        
        if desc_list is None:
            return out
        
        for name, func in desc_list:
            try:
                val = func(mol)
                if isinstance(val, float) and np.isnan(val):
                    out[name] = self.missing_val
                else:
                    out[name] = val
            except Exception:
                out[name] = self.missing_val
        
        return out
    
    def compute_from_smiles(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Calculate descriptors from SMILES string."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.compute(mol)


class MolecularFeaturizer:
    """
    Complete molecular featurization pipeline.
    
    Computes Morgan fingerprints and RDKit descriptors for a dataset.
    """
    
    def __init__(
        self,
        fp_n_bits: int = FP_NBITS,
        fp_radius: int = FP_RADIUS,
        compute_fingerprints: bool = True,
        compute_descriptors: bool = True
    ):
        """
        Initialize featurizer.
        
        Args:
            fp_n_bits: Fingerprint bit count
            fp_radius: Fingerprint radius
            compute_fingerprints: Whether to compute fingerprints
            compute_descriptors: Whether to compute RDKit descriptors
        """
        self.fp_calculator = MorganFingerprintCalculator(fp_n_bits, fp_radius)
        self.desc_calculator = RDKitDescriptorCalculator()
        self.compute_fingerprints = compute_fingerprints
        self.compute_descriptors = compute_descriptors
    
    def featurize_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "canonical_smiles",
        preserve_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Featurize all molecules in a DataFrame.
        
        Args:
            df: Input DataFrame with SMILES
            smiles_col: Name of SMILES column
            preserve_cols: Columns to preserve in output
            
        Returns:
            DataFrame with features added
        """
        from rdkit import Chem
        
        if preserve_cols is None:
            preserve_cols = [
                'Drug_ID', 'original_smiles', 'Y', 'task_name',
                'source', 'task', 'canonical_smiles'
            ]
        
        rows = []
        n_failed = 0
        
        for idx, row in df.iterrows():
            smiles = row.get(smiles_col)
            if pd.isna(smiles):
                n_failed += 1
                continue
            
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                n_failed += 1
                continue
            
            # Start with preserved columns
            feat_row = {
                col: row.get(col) for col in preserve_cols if col in row.index
            }
            
            # Add fingerprints
            if self.compute_fingerprints:
                fp_bits, fp_packed = self.fp_calculator.compute(mol)
                feat_row['morgan_fp_bits'] = fp_bits
                feat_row['morgan_fp_packed'] = fp_packed
            
            # Add descriptors
            if self.compute_descriptors:
                descriptors = self.desc_calculator.compute(mol)
                feat_row.update(descriptors)
            
            rows.append(feat_row)
        
        logger.info(
            f"Featurized {len(rows)} molecules, {n_failed} failed"
        )
        
        return pd.DataFrame(rows)
    
    def save_parquet(self, df: pd.DataFrame, path: Path) -> Path:
        """Save featurized data to Parquet format."""
        path = Path(path)
        df.to_parquet(path, index=False)
        logger.info(f"Saved features to {path}")
        return path


def impute_missing_descriptors(
    df: pd.DataFrame,
    descriptor_cols: List[str] = None,
    strategy: str = "median"
) -> pd.DataFrame:
    """
    Impute missing values in descriptor columns.
    
    Args:
        df: DataFrame with descriptors
        descriptor_cols: Columns to impute (auto-detected if None)
        strategy: Imputation strategy ('median', 'mean', 'zero')
        
    Returns:
        DataFrame with imputed values
    """
    from sklearn.impute import SimpleImputer
    
    df = df.copy()
    
    # Auto-detect numeric descriptor columns
    if descriptor_cols is None:
        exclude_cols = {
            'Drug_ID', 'original_smiles', 'canonical_smiles',
            'task_name', 'source', 'task', 'Y',
            'morgan_fp_bits', 'morgan_fp_packed'
        }
        descriptor_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64']
        ]
    
    if not descriptor_cols:
        return df
    
    if strategy == "zero":
        df[descriptor_cols] = df[descriptor_cols].fillna(0)
    else:
        imputer = SimpleImputer(strategy=strategy)
        df[descriptor_cols] = imputer.fit_transform(df[descriptor_cols])
    
    logger.info(f"Imputed {len(descriptor_cols)} descriptor columns using {strategy}")
    
    return df
