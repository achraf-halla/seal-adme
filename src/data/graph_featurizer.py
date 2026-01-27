"""
Graph featurization for SEAL-ADME.

This module converts molecules to PyTorch Geometric graph objects with:
- Rich atom features (type, degree, charge, aromaticity, etc.)
- Gasteiger partial charges
- BRICS fragment assignments
- Inter/intra-fragment edge masks
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_ATOM_TYPE_SET,
    DEFAULT_HYBRIDIZATION_SET,
    PAULING_ELECTRONEGATIVITY,
    DEFAULT_ELECTRONEGATIVITY,
)
from .fragmentation import (
    FragmentExtractor,
    brics_decomp_extra,
    extract_fragment_metadata,
    create_cluster_assignment_matrix,
    create_atom_wise_assignment_matrix,
    mask_broken_edges,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Atom featurization utilities
# =============================================================================

def one_hot_encoding(x, allowable_set: List, with_unknown: bool = True) -> List[float]:
    """
    Create one-hot encoding for a value.
    
    Args:
        x: Value to encode
        allowable_set: List of allowed values
        with_unknown: If True, last position is reserved for unknown values
        
    Returns:
        One-hot encoded list
    """
    if with_unknown and x not in allowable_set:
        x = allowable_set[-1]  # Use last element as "unknown"
    return [float(x == s) for s in allowable_set]


class AtomFeaturizer:
    """
    Compute atom-level features for molecules.
    
    Features include:
    - Atom type (one-hot)
    - Hybridization (one-hot)
    - Degree
    - Formal charge
    - Aromaticity
    - Ring membership
    - Number of hydrogens
    - Implicit valence
    - Numeric features (atomic number, mass, electronegativity, Gasteiger charge)
    """
    
    def __init__(
        self,
        atom_types: List[str] = DEFAULT_ATOM_TYPE_SET,
        hybridization_types: List[str] = DEFAULT_HYBRIDIZATION_SET,
        include_degree: bool = True,
        include_charge: bool = True,
        include_aromatic: bool = True,
        include_hybridization: bool = True,
        include_valence: bool = True,
        include_ring: bool = True,
        include_num_hs: bool = True,
        include_numeric: bool = True,
        include_gasteiger: bool = True,
    ):
        """
        Initialize atom featurizer.
        
        Args:
            atom_types: List of atom types for one-hot encoding
            hybridization_types: List of hybridization types
            include_*: Flags to include various feature types
        """
        self.atom_types = atom_types
        self.hybridization_types = hybridization_types
        self.include_degree = include_degree
        self.include_charge = include_charge
        self.include_aromatic = include_aromatic
        self.include_hybridization = include_hybridization
        self.include_valence = include_valence
        self.include_ring = include_ring
        self.include_num_hs = include_num_hs
        self.include_numeric = include_numeric
        self.include_gasteiger = include_gasteiger
    
    def get_atom_type(self, atom) -> List[float]:
        """One-hot encoding of atom type."""
        return one_hot_encoding(atom.GetSymbol(), self.atom_types)
    
    def get_hybridization(self, atom) -> List[float]:
        """One-hot encoding of hybridization."""
        hyb = str(atom.GetHybridization()).replace('.', '').upper()
        return one_hot_encoding(hyb, self.hybridization_types)
    
    def get_degree(self, atom) -> List[float]:
        """Atom degree (number of bonds)."""
        return [float(atom.GetDegree())]
    
    def get_formal_charge(self, atom) -> List[float]:
        """Formal charge."""
        return [float(atom.GetFormalCharge())]
    
    def get_is_aromatic(self, atom) -> List[float]:
        """Whether atom is aromatic."""
        return [1.0 if atom.GetIsAromatic() else 0.0]
    
    def get_is_in_ring(self, atom) -> List[float]:
        """Whether atom is in a ring."""
        return [1.0 if atom.IsInRing() else 0.0]
    
    def get_num_hs(self, atom) -> List[float]:
        """Number of hydrogens (including implicit)."""
        return [float(atom.GetTotalNumHs())]
    
    def get_valence(self, atom) -> List[float]:
        """Implicit valence."""
        return [float(atom.GetImplicitValence())]
    
    def get_numeric_features(
        self, 
        atom, 
        gasteiger_charge: Optional[float] = None
    ) -> List[float]:
        """
        Scaled numeric features.
        
        Returns normalized:
        - Atomic number / 100
        - Atomic mass / 200
        - Electronegativity / 4
        - Gasteiger charge
        """
        atomic_num = float(atom.GetAtomicNum()) / 100.0
        atom_mass = float(atom.GetMass()) / 200.0
        
        symbol = atom.GetSymbol()
        elneg = PAULING_ELECTRONEGATIVITY.get(symbol, DEFAULT_ELECTRONEGATIVITY) / 4.0
        
        g_charge = float(gasteiger_charge) if gasteiger_charge is not None else 0.0
        
        return [atomic_num, atom_mass, elneg, g_charge]
    
    def featurize_atom(
        self,
        atom,
        gasteiger_charge: Optional[float] = None,
    ) -> List[float]:
        """
        Compute all features for a single atom.
        
        Args:
            atom: RDKit atom object
            gasteiger_charge: Pre-computed Gasteiger charge
            
        Returns:
            List of features
        """
        features = []
        
        # Base features (atom type)
        features.extend(self.get_atom_type(atom))
        
        # Optional features
        if self.include_degree:
            features.extend(self.get_degree(atom))
        
        if self.include_charge:
            features.extend(self.get_formal_charge(atom))
        
        if self.include_aromatic:
            features.extend(self.get_is_aromatic(atom))
        
        if self.include_hybridization:
            features.extend(self.get_hybridization(atom))
        
        if self.include_valence:
            features.extend(self.get_valence(atom))
        
        if self.include_ring:
            features.extend(self.get_is_in_ring(atom))
        
        if self.include_num_hs:
            features.extend(self.get_num_hs(atom))
        
        if self.include_numeric:
            features.extend(self.get_numeric_features(atom, gasteiger_charge))
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the total dimension of atom features."""
        dim = len(self.atom_types)  # Base atom type
        
        if self.include_degree:
            dim += 1
        if self.include_charge:
            dim += 1
        if self.include_aromatic:
            dim += 1
        if self.include_hybridization:
            dim += len(self.hybridization_types)
        if self.include_valence:
            dim += 1
        if self.include_ring:
            dim += 1
        if self.include_num_hs:
            dim += 1
        if self.include_numeric:
            dim += 4  # atomic_num, mass, elneg, gasteiger
        
        return dim


# =============================================================================
# Gasteiger charge computation
# =============================================================================

def compute_gasteiger_charges(mol) -> np.ndarray:
    """
    Compute Gasteiger partial charges for all atoms.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Array of charges, shape (num_atoms,)
    """
    try:
        import rdkit.Chem.rdPartialCharges as rdPartialCharges
    except ImportError:
        raise ImportError("RDKit not installed")
    
    num_atoms = mol.GetNumAtoms()
    
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp('_GasteigerCharge'):
                c = atom.GetProp('_GasteigerCharge')
                try:
                    charges.append(float(c))
                except (ValueError, TypeError):
                    charges.append(0.0)
            else:
                charges.append(0.0)
        return np.array(charges, dtype=np.float32)
    except Exception:
        return np.zeros(num_atoms, dtype=np.float32)


# =============================================================================
# Edge extraction
# =============================================================================

def get_edge_index(mol) -> np.ndarray:
    """
    Extract edge index from molecule (COO format).
    
    Creates undirected edges (both directions for each bond).
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Edge index array of shape (2, num_edges)
    """
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions for undirected graph
        edges.append([i, j])
        edges.append([j, i])
    
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    
    edges_arr = np.array(edges, dtype=np.int64).T
    
    # Sort by source, then destination for consistency
    sort_idx = np.lexsort((edges_arr[1], edges_arr[0]))
    return edges_arr[:, sort_idx]


# =============================================================================
# Main GraphFeaturizer class
# =============================================================================

class GraphFeaturizer:
    """
    Convert molecules to PyTorch Geometric graph objects.
    
    Creates graphs with:
    - Node features (atom features)
    - Edge index (molecular bonds)
    - Cluster assignment matrices (S, S_node)
    - Inter-fragment edge masks
    - Fragment metadata (optional)
    """
    
    def __init__(
        self,
        y_column: str = "Y",
        smiles_col: str = "canonical_smiles",
        include_optional: bool = True,
        include_numeric: bool = True,
        include_gasteiger: bool = True,
        store_fragments: bool = True,
    ):
        """
        Initialize graph featurizer.
        
        Args:
            y_column: Name of target column
            smiles_col: Name of SMILES column
            include_optional: Include optional atom features
            include_numeric: Include numeric atom features
            include_gasteiger: Include Gasteiger charges
            store_fragments: Store fragment metadata in graphs
        """
        self.y_column = y_column
        self.smiles_col = smiles_col
        self.store_fragments = store_fragments
        
        # Initialize atom featurizer
        self.atom_featurizer = AtomFeaturizer(
            include_degree=include_optional,
            include_charge=include_optional,
            include_aromatic=include_optional,
            include_hybridization=include_optional,
            include_valence=include_optional,
            include_ring=include_optional,
            include_num_hs=include_optional,
            include_numeric=include_numeric,
            include_gasteiger=include_gasteiger,
        )
        
        self.fragment_extractor = FragmentExtractor()
    
    def featurize_molecule(
        self,
        smiles: str,
        y_value: float,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        drug_id: Optional[str] = None,
    ):
        """
        Convert a single molecule to a graph.
        
        Args:
            smiles: SMILES string
            y_value: Target value
            y_mean: Mean for normalization
            y_std: Std for normalization
            drug_id: Optional molecule identifier
            
        Returns:
            PyTorch Geometric Data object, or None if featurization fails
        """
        try:
            from rdkit import Chem
            import torch
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError(f"Required library not installed: {e}")
        
        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles[:50]}...")
            return None
        
        # Normalize target
        y_norm = (y_value - y_mean) / y_std if y_std != 0 else y_value
        
        # Compute Gasteiger charges
        g_charges = compute_gasteiger_charges(mol)
        
        # Extract edges
        edge_index = get_edge_index(mol)
        edge_index = torch.LongTensor(edge_index)
        
        # Compute atom features
        nodes = []
        for i, atom in enumerate(mol.GetAtoms()):
            gc = g_charges[i] if i < len(g_charges) else 0.0
            features = self.atom_featurizer.featurize_atom(atom, gasteiger_charge=gc)
            nodes.append(features)
        
        if nodes:
            x = torch.FloatTensor(np.array(nodes))
            # Handle NaN/Inf values
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            # Empty molecule fallback
            feat_dim = self.atom_featurizer.get_feature_dim()
            x = torch.empty((0, feat_dim), dtype=torch.float32)
        
        # BRICS decomposition
        num_atoms = mol.GetNumAtoms()
        cliques, breaks = brics_decomp_extra(mol)
        
        if cliques is None or len(cliques) == 0:
            cliques = [[i] for i in range(num_atoms)]
            breaks = []
        
        # Cluster assignment matrices
        s = create_cluster_assignment_matrix(cliques, num_atoms)
        s_node = create_atom_wise_assignment_matrix(cliques, num_atoms)
        
        s = torch.FloatTensor(s)
        s_node = torch.FloatTensor(s_node)
        
        # Inter-fragment edge mask
        if edge_index.numel() != 0:
            mask = mask_broken_edges(edge_index.numpy(), breaks)
            mask = torch.FloatTensor(mask)
        else:
            mask = torch.empty((0,), dtype=torch.float32)
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            s=s,
            s_node=s_node,
            y=torch.FloatTensor([float(y_norm)]),
            num_cluster=torch.LongTensor([int(s.shape[1])]),
            mask=mask,
        )
        
        # Store metadata
        data.drug_id = drug_id
        data.canonical_smiles = smiles
        
        # Store fragment information
        if self.store_fragments:
            fragment_meta = extract_fragment_metadata(mol, cliques)
            data.fragment_smiles = fragment_meta['fragment_smiles']
            data.fragment_atom_lists = fragment_meta['fragment_atom_lists']
            data.fragment_sizes = fragment_meta['fragment_sizes']
            if 'fragment_fingerprints' in fragment_meta:
                data.fragment_fingerprints = fragment_meta['fragment_fingerprints']
        
        return data
    
    def __call__(
        self,
        df: pd.DataFrame,
        stats: Dict[str, float],
    ) -> List:
        """
        Featurize all molecules in a DataFrame.
        
        Args:
            df: DataFrame with molecules
            stats: Dictionary with "mean" and "std" for normalization
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        import torch
        
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)
        if std == 0:
            std = 1.0
        
        graphs = []
        failed = 0
        
        for idx, row in df.iterrows():
            smiles = row.get(self.smiles_col)
            y_val = row.get(self.y_column, 0.0)
            drug_id = row.get("Drug_ID")
            
            if pd.isna(smiles):
                failed += 1
                continue
            
            try:
                y_val = float(y_val) if not pd.isna(y_val) else 0.0
            except (ValueError, TypeError):
                y_val = 0.0
            
            data = self.featurize_molecule(
                smiles=smiles,
                y_value=y_val,
                y_mean=mean,
                y_std=std,
                drug_id=drug_id,
            )
            
            if data is not None:
                graphs.append(data)
            else:
                failed += 1
        
        if failed > 0:
            logger.warning(f"Failed to featurize {failed}/{len(df)} molecules")
        
        # Pad cluster matrices to same size for batching
        if graphs:
            graphs = self._pad_cluster_matrices(graphs)
        
        return graphs
    
    def _pad_cluster_matrices(self, graphs: List) -> List:
        """
        Pad cluster assignment matrices to consistent sizes.
        
        Required for batching graphs with different numbers of clusters.
        """
        import torch
        
        if not graphs:
            return graphs
        
        max_clusters = max(d.s.size(1) for d in graphs)
        max_nodes = max(d.s_node.size(1) for d in graphs)
        
        for d in graphs:
            # Pad S matrix
            if d.s.size(1) < max_clusters:
                pad = torch.zeros(
                    (d.s.size(0), max_clusters - d.s.size(1)),
                    dtype=d.s.dtype
                )
                d.s = torch.cat([d.s, pad], dim=1)
            
            # Pad S_node matrix
            if d.s_node.size(1) < max_nodes:
                pad = torch.zeros(
                    (d.s_node.size(0), max_nodes - d.s_node.size(1)),
                    dtype=d.s_node.dtype
                )
                d.s_node = torch.cat([d.s_node, pad], dim=1)
        
        return graphs


# =============================================================================
# Batch processing utilities
# =============================================================================

def featurize_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    task_name: str,
    y_column: str = "Y",
    smiles_col: str = "canonical_smiles",
    compute_stats_from: Optional[pd.DataFrame] = None,
    store_fragments: bool = True,
) -> Dict:
    """
    Featurize a dataset and save graphs to disk.
    
    Args:
        df: DataFrame with molecules
        output_dir: Directory to save graph files
        task_name: Name of the task/dataset
        y_column: Name of target column
        smiles_col: Name of SMILES column
        compute_stats_from: DataFrame to compute normalization stats from
        store_fragments: Whether to store fragment metadata
        
    Returns:
        Dictionary with processing statistics
    """
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute normalization statistics
    stats_df = compute_stats_from if compute_stats_from is not None else df
    mean = float(stats_df[y_column].mean())
    std = float(stats_df[y_column].std())
    if std == 0:
        std = 1.0
    
    stats = {"mean": mean, "std": std}
    
    # Initialize featurizer
    featurizer = GraphFeaturizer(
        y_column=y_column,
        smiles_col=smiles_col,
        include_optional=True,
        include_numeric=True,
        include_gasteiger=True,
        store_fragments=store_fragments,
    )
    
    # Featurize all molecules
    graphs = featurizer(df, stats)
    
    # Save individual graph files
    for i, graph in enumerate(graphs):
        filename = output_dir / f"{task_name}_{i:06d}.pt"
        torch.save(graph, filename)
    
    logger.info(f"Saved {len(graphs)} graphs to {output_dir}")
    
    return {
        "n_graphs": len(graphs),
        "mean": mean,
        "std": std,
        "output_dir": str(output_dir),
    }
