"""
PyTorch Geometric graph featurization with SEAL-style fragment awareness.

This module creates molecular graphs with:
- Node features (atom properties)
- Edge indices (bonds)
- Fragment membership matrices (S matrix for SEAL)
- Fragment metadata for interpretability
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .constants import (
    DEFAULT_ATOM_TYPE_SET,
    DEFAULT_HYBRIDIZATION_SET,
    PAULING_ELECTRONEGATIVITY,
    DEFAULT_ELECTRONEGATIVITY,
    GRAPH_COLUMNS
)
from .fragmentation import brics_decompose, FragmentExtractor

logger = logging.getLogger(__name__)


def one_of_k_encoding_unk(x: Any, allowable_set: List) -> List[float]:
    """
    One-hot encoding with unknown category.
    
    Args:
        x: Value to encode
        allowable_set: List of allowed values (last is 'unknown')
        
    Returns:
        One-hot encoded list
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [float(x == s) for s in allowable_set]


class AtomFeaturizer:
    """
    Compute atom-level features for graph nodes.
    
    Features include:
    - Atom type (one-hot)
    - Degree
    - Formal charge
    - Aromaticity
    - Hybridization (one-hot)
    - Implicit valence
    - Ring membership
    - Number of hydrogens
    - Atomic number (scaled)
    - Atomic mass (scaled)
    - Electronegativity (scaled)
    - Gasteiger charge
    """
    
    def __init__(
        self,
        include_optional: bool = True,
        include_numeric: bool = True,
        include_gasteiger: bool = True
    ):
        """
        Initialize atom featurizer.
        
        Args:
            include_optional: Include degree, charge, aromaticity, etc.
            include_numeric: Include scaled numeric features
            include_gasteiger: Compute Gasteiger partial charges
        """
        self.include_optional = include_optional
        self.include_numeric = include_numeric
        self.include_gasteiger = include_gasteiger
    
    def compute_gasteiger_charges(self, mol) -> np.ndarray:
        """Compute Gasteiger partial charges for all atoms."""
        import rdkit.Chem.rdPartialCharges as rdPartialCharges
        
        if not self.include_gasteiger:
            return np.zeros((mol.GetNumAtoms(),), dtype=float)
        
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                if atom.HasProp('_GasteigerCharge'):
                    try:
                        charges.append(float(atom.GetProp('_GasteigerCharge')))
                    except (ValueError, TypeError):
                        charges.append(0.0)
                else:
                    charges.append(0.0)
            return np.array(charges, dtype=float)
        except Exception:
            return np.zeros((mol.GetNumAtoms(),), dtype=float)
    
    def featurize_atom(
        self,
        atom,
        gasteiger_charge: float = 0.0
    ) -> List[float]:
        """
        Compute features for a single atom.
        
        Args:
            atom: RDKit Atom object
            gasteiger_charge: Precomputed Gasteiger charge
            
        Returns:
            List of float features
        """
        feats = []
        
        # Atom type (one-hot)
        feats.extend(one_of_k_encoding_unk(
            atom.GetSymbol(), DEFAULT_ATOM_TYPE_SET
        ))
        
        if self.include_optional:
            # Degree
            feats.append(float(atom.GetDegree()))
            # Formal charge
            feats.append(float(atom.GetFormalCharge()))
            # Aromaticity
            feats.append(1.0 if atom.GetIsAromatic() else 0.0)
            # Hybridization (one-hot)
            hyb = str(atom.GetHybridization()).replace('.', '').upper()
            feats.extend(one_of_k_encoding_unk(hyb, DEFAULT_HYBRIDIZATION_SET))
            # Implicit valence
            feats.append(float(atom.GetImplicitValence()))
            # Ring membership
            feats.append(1.0 if atom.IsInRing() else 0.0)
            # Number of hydrogens
            feats.append(float(atom.GetTotalNumHs()))
        
        if self.include_numeric:
            # Scaled atomic number
            feats.append(float(atom.GetAtomicNum()) / 100.0)
            # Scaled atomic mass
            feats.append(float(atom.GetMass()) / 200.0)
            # Scaled electronegativity
            en = PAULING_ELECTRONEGATIVITY.get(
                atom.GetSymbol(), DEFAULT_ELECTRONEGATIVITY
            )
            feats.append(float(en) / 4.0)
            # Gasteiger charge
            feats.append(float(gasteiger_charge))
        
        return feats
    
    @property
    def feature_dim(self) -> int:
        """Get the dimension of atom feature vector."""
        from rdkit import Chem
        
        # Use a dummy atom to compute feature dimension
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)
        return len(self.featurize_atom(atom, 0.0))


class GraphFeaturizer:
    """
    Create PyTorch Geometric Data objects with SEAL-style features.
    
    Each graph contains:
    - x: Node features [num_atoms, feature_dim]
    - edge_index: Edge indices [2, num_edges]
    - s: Fragment membership matrix [num_atoms, num_fragments]
    - s_node: Atom identity matrix [num_atoms, num_atoms]
    - mask: Edge mask for broken bonds [num_edges]
    - y: Target value [1]
    - Fragment metadata for interpretability
    """
    
    def __init__(
        self,
        y_column: str = 'Y',
        smiles_col: str = 'canonical_smiles',
        include_optional: bool = True,
        include_numeric: bool = True,
        include_gasteiger: bool = True,
        store_fragments: bool = True
    ):
        """
        Initialize graph featurizer.
        
        Args:
            y_column: Column name for target values
            smiles_col: Column name for SMILES
            include_optional: Include optional atom features
            include_numeric: Include scaled numeric features
            include_gasteiger: Include Gasteiger charges
            store_fragments: Store fragment metadata in graphs
        """
        self.y_column = y_column
        self.smiles_col = smiles_col
        self.store_fragments = store_fragments
        
        self.atom_featurizer = AtomFeaturizer(
            include_optional=include_optional,
            include_numeric=include_numeric,
            include_gasteiger=include_gasteiger
        )
        self.fragment_extractor = FragmentExtractor()
    
    def _create_fragment_matrix(
        self,
        cliques: List[List[int]],
        num_atoms: int
    ) -> torch.Tensor:
        """Create fragment membership matrix S."""
        s = torch.zeros((num_atoms, len(cliques)), dtype=torch.float32)
        for clique_idx, clique in enumerate(cliques):
            for atom_idx in clique:
                s[atom_idx, clique_idx] = 1.0
        return s
    
    def _create_edge_mask(
        self,
        edge_index: torch.Tensor,
        breaks: List[List[int]]
    ) -> torch.Tensor:
        """Create mask for edges (0 for broken bonds, 1 otherwise)."""
        if edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)
        
        num_edges = edge_index.size(1)
        mask = torch.ones(num_edges, dtype=torch.float32)
        
        broken_edges = {frozenset(b) for b in breaks}
        for i in range(num_edges):
            edge = frozenset([
                edge_index[0, i].item(),
                edge_index[1, i].item()
            ])
            if edge in broken_edges:
                mask[i] = 0.0
        
        return mask
    
    def _extract_fragment_metadata(
        self,
        mol,
        cliques: List[List[int]]
    ) -> Dict[str, List]:
        """Extract fragment information for interpretability."""
        metadata = {
            'fragment_smiles': [],
            'fragment_atom_lists': [],
            'fragment_sizes': [],
            'fragment_fingerprints': []
        }
        
        for clique in cliques:
            frag_smiles = self.fragment_extractor.extract_fragment_smiles(mol, clique)
            metadata['fragment_smiles'].append(frag_smiles)
            metadata['fragment_atom_lists'].append(clique)
            metadata['fragment_sizes'].append(len(clique))
            
            if frag_smiles:
                fp = self.fragment_extractor.compute_fragment_fingerprint(frag_smiles)
                metadata['fragment_fingerprints'].append(fp)
            else:
                metadata['fragment_fingerprints'].append(None)
        
        return metadata
    
    def featurize_molecule(
        self,
        smiles: str,
        y: float,
        mean: float = 0.0,
        std: float = 1.0,
        drug_id: Any = None
    ) -> Optional[Data]:
        """
        Create a PyG Data object from a single molecule.
        
        Args:
            smiles: SMILES string
            y: Target value
            mean: Mean for target normalization
            std: Std for target normalization
            drug_id: Optional identifier
            
        Returns:
            PyG Data object or None if featurization fails
        """
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Normalize target
        y_norm = (float(y) - mean) / std if std != 0 else float(y) - mean
        
        # Compute Gasteiger charges
        g_charges = self.atom_featurizer.compute_gasteiger_charges(mol)
        
        # Build edge index
        edges = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edges.append((start, end))
            edges.append((end, start))
        
        if edges:
            edge_index = torch.LongTensor(np.array(edges).T)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Build node features
        nodes = []
        for i, atom in enumerate(mol.GetAtoms()):
            gc = g_charges[i] if i < len(g_charges) else 0.0
            feats = self.atom_featurizer.featurize_atom(atom, gasteiger_charge=gc)
            nodes.append(feats)
        
        if nodes:
            x = torch.FloatTensor(np.array(nodes))
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            x = torch.empty((0, self.atom_featurizer.feature_dim), dtype=torch.float32)
        
        # BRICS fragmentation
        num_atoms = mol.GetNumAtoms()
        cliques, breaks = brics_decompose(mol)
        
        if not cliques:
            cliques = [[i] for i in range(num_atoms)]
        
        # Create fragment matrices
        s = self._create_fragment_matrix(cliques, num_atoms)
        s_node = torch.eye(num_atoms, dtype=torch.float32)
        mask = self._create_edge_mask(edge_index, breaks)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            s=s,
            s_node=s_node,
            y=torch.FloatTensor([y_norm]),
            num_cluster=torch.LongTensor([s.shape[1]]),
            mask=mask
        )
        
        # Store metadata
        data.drug_id = drug_id
        data.canonical_smiles = smiles
        
        # Store fragment information
        if self.store_fragments:
            fragment_meta = self._extract_fragment_metadata(mol, cliques)
            data.fragment_smiles = fragment_meta['fragment_smiles']
            data.fragment_atom_lists = fragment_meta['fragment_atom_lists']
            data.fragment_sizes = fragment_meta['fragment_sizes']
            data.fragment_fingerprints = fragment_meta['fragment_fingerprints']
        
        return data
    
    def __call__(
        self,
        df: pd.DataFrame,
        stats: Dict[str, float] = None
    ) -> List[Data]:
        """
        Featurize all molecules in a DataFrame.
        
        Args:
            df: DataFrame with SMILES and targets
            stats: Dict with 'mean' and 'std' for normalization
            
        Returns:
            List of PyG Data objects
        """
        if stats is None:
            stats = {}
        
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)
        
        graphs = []
        n_failed = 0
        
        for idx, row in df.iterrows():
            try:
                y = float(row[self.y_column])
            except (ValueError, TypeError):
                n_failed += 1
                continue
            
            smiles = row.get(self.smiles_col)
            if pd.isna(smiles):
                n_failed += 1
                continue
            
            data = self.featurize_molecule(
                smiles=smiles,
                y=y,
                mean=mean,
                std=std,
                drug_id=row.get("Drug_ID")
            )
            
            if data is not None:
                graphs.append(data)
            else:
                n_failed += 1
        
        logger.info(f"Created {len(graphs)} graphs, {n_failed} failed")
        
        # Pad fragment matrices to consistent size
        if graphs:
            graphs = self._pad_fragment_matrices(graphs)
        
        return graphs
    
    def _pad_fragment_matrices(self, graphs: List[Data]) -> List[Data]:
        """Pad S matrices to maximum cluster count across batch."""
        max_clusters = max(d.s.size(1) for d in graphs)
        max_nodes = max(d.s_node.size(1) for d in graphs)
        
        for d in graphs:
            if d.s.size(1) < max_clusters:
                pad = torch.zeros(
                    (d.s.size(0), max_clusters - d.s.size(1)),
                    dtype=d.s.dtype
                )
                d.s = torch.cat([d.s, pad], dim=1)
            
            if d.s_node.size(1) < max_nodes:
                pad = torch.zeros(
                    (d.s_node.size(0), max_nodes - d.s_node.size(1)),
                    dtype=d.s_node.dtype
                )
                d.s_node = torch.cat([d.s_node, pad], dim=1)
        
        return graphs


def save_graphs(
    graphs: List[Data],
    output_dir: Path,
    prefix: str = "graph"
) -> List[Path]:
    """
    Save graphs to individual .pt files.
    
    Args:
        graphs: List of PyG Data objects
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, g in enumerate(graphs):
        path = output_dir / f"{prefix}_{i:05d}.pt"
        torch.save(g, path)
        paths.append(path)
    
    logger.info(f"Saved {len(paths)} graphs to {output_dir}")
    return paths


def load_graphs(input_dir: Path, pattern: str = "*.pt") -> List[Data]:
    """
    Load graphs from .pt files.
    
    Args:
        input_dir: Directory containing .pt files
        pattern: Glob pattern for files
        
    Returns:
        List of PyG Data objects
    """
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob(pattern))
    
    graphs = []
    for path in paths:
        graphs.append(torch.load(path))
    
    logger.info(f"Loaded {len(graphs)} graphs from {input_dir}")
    return graphs
