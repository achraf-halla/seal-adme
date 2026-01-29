"""
PyTorch Geometric graph construction for molecular data.

Creates graph representations with:
- Node features based on atom properties
- Edge connectivity from bonds
- Fragment membership matrices from BRICS decomposition
- Edge masks for intra/inter-fragment message passing
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .constants import (
    ATOM_TYPE_SET,
    HYBRIDIZATION_SET,
    PAULING_ELECTRONEGATIVITY,
    DEFAULT_ELECTRONEGATIVITY,
)
from .fragmentation import (
    brics_decompose,
    get_fragment_membership_matrix,
    get_edge_break_mask,
    FragmentExtractor,
)

logger = logging.getLogger(__name__)


def one_hot_encode(value: Any, vocab: List[Any]) -> List[int]:
    """One-hot encode a value given a vocabulary."""
    encoding = [0] * len(vocab)
    if value in vocab:
        encoding[vocab.index(value)] = 1
    else:
        # Unknown value - encode as last position if "Unk" exists
        if "Unk" in vocab:
            encoding[vocab.index("Unk")] = 1
        elif "Other" in vocab:
            encoding[vocab.index("Other")] = 1
    return encoding


def get_atom_features(atom) -> List[float]:
    """
    Extract features for a single atom.
    
    Features (25 total):
    - Atom type one-hot (11)
    - Degree one-hot (6: 0-5)
    - Hybridization one-hot (4)
    - Is aromatic (1)
    - Is in ring (1)
    - Electronegativity (1)
    - Formal charge (1)
    
    Args:
        atom: RDKit Atom object
        
    Returns:
        List of atom features
    """
    features = []
    
    # Atom type (11)
    symbol = atom.GetSymbol()
    features.extend(one_hot_encode(symbol, ATOM_TYPE_SET))
    
    # Degree (6)
    degree = min(atom.GetDegree(), 5)
    features.extend(one_hot_encode(degree, list(range(6))))
    
    # Hybridization (4)
    hyb = str(atom.GetHybridization())
    hyb_clean = hyb.replace("HybridizationType.", "")
    features.extend(one_hot_encode(hyb_clean, HYBRIDIZATION_SET))
    
    # Aromatic (1)
    features.append(float(atom.GetIsAromatic()))
    
    # In ring (1)
    features.append(float(atom.IsInRing()))
    
    # Electronegativity (1) - normalized
    en = PAULING_ELECTRONEGATIVITY.get(symbol, DEFAULT_ELECTRONEGATIVITY)
    features.append(en / 4.0)  # Normalize by max (F = 3.98)
    
    # Formal charge (1)
    features.append(float(atom.GetFormalCharge()))
    
    return features


def mol_to_graph(
    mol,
    y_value: float = 0.0,
    normalize_y: bool = False,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    drug_id: str = "",
    task_name: str = "",
    smiles: str = "",
    store_fragments: bool = True
) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric Data object.
    
    Args:
        mol: RDKit Mol object
        y_value: Target value
        normalize_y: Whether to normalize y
        y_mean: Mean for normalization
        y_std: Std for normalization
        drug_id: Drug identifier
        task_name: Name of the task
        smiles: Canonical SMILES
        store_fragments: Whether to extract fragment SMILES
        
    Returns:
        PyG Data object with node features, edge index, fragment info
    """
    # Get atom features
    n_atoms = mol.GetNumAtoms()
    x = []
    for atom in mol.GetAtoms():
        x.append(get_atom_features(atom))
    x = torch.tensor(x, dtype=torch.float32)
    
    # Get edge index
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append([i, j])
        edge_list.append([j, i])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # BRICS decomposition
    cliques, breaks = brics_decompose(mol)
    
    # Fragment membership matrix S
    S = get_fragment_membership_matrix(n_atoms, cliques)
    s = torch.tensor(S, dtype=torch.float32)
    
    # Edge mask (True = within fragment)
    edge_tuples = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    mask_list = get_edge_break_mask(edge_tuples, breaks)
    mask = torch.tensor(mask_list, dtype=torch.float32)
    
    # Target
    if normalize_y and y_std > 0:
        y_normalized = (y_value - y_mean) / y_std
    else:
        y_normalized = y_value
    y = torch.tensor([y_normalized], dtype=torch.float32)
    
    # Create data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.s = s
    data.mask = mask
    data.drug_id = drug_id
    data.task_name = task_name
    data.smiles = smiles
    data.n_fragments = len(cliques)
    
    # Store fragment info
    if store_fragments:
        fragment_info = FragmentExtractor.extract_all_fragments(mol, cliques)
        data.fragment_smiles = [f['smiles'] for f in fragment_info]
        data.fragment_atom_lists = [f['atom_indices'] for f in fragment_info]
        data.fragment_sizes = [f['size'] for f in fragment_info]
    
    return data


class GraphBuilder:
    """
    Build PyG graphs from SMILES strings.
    
    Handles batch processing with optional normalization.
    """
    
    def __init__(
        self,
        y_column: str = "Y",
        smiles_col: str = "canonical_smiles",
        drug_id_col: str = "Drug_ID",
        task_name_col: str = "task_name",
        store_fragments: bool = True,
        normalize_y: bool = False
    ):
        """
        Initialize graph builder.
        
        Args:
            y_column: Column containing target values
            smiles_col: Column containing SMILES
            drug_id_col: Column containing drug IDs
            task_name_col: Column containing task names
            store_fragments: Whether to store fragment SMILES
            normalize_y: Whether to normalize targets
        """
        self.y_column = y_column
        self.smiles_col = smiles_col
        self.drug_id_col = drug_id_col
        self.task_name_col = task_name_col
        self.store_fragments = store_fragments
        self.normalize_y = normalize_y
    
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        y_mean: float = None,
        y_std: float = None
    ) -> List[Data]:
        """
        Build graphs from DataFrame.
        
        Args:
            df: DataFrame with SMILES and targets
            y_mean: Mean for normalization (computed if None)
            y_std: Std for normalization (computed if None)
            
        Returns:
            List of PyG Data objects
        """
        from rdkit import Chem
        
        # Compute normalization stats if needed
        if self.normalize_y:
            if y_mean is None:
                y_mean = df[self.y_column].mean()
            if y_std is None:
                y_std = df[self.y_column].std()
                if y_std == 0:
                    y_std = 1.0
        else:
            y_mean = 0.0
            y_std = 1.0
        
        graphs = []
        n_failed = 0
        
        for idx, row in df.iterrows():
            smiles = row.get(self.smiles_col, "")
            if pd.isna(smiles) or smiles == "":
                n_failed += 1
                continue
            
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                n_failed += 1
                continue
            
            y_value = float(row.get(self.y_column, 0.0))
            drug_id = str(row.get(self.drug_id_col, f"mol_{len(graphs)}"))
            task_name = str(row.get(self.task_name_col, ""))
            
            try:
                data = mol_to_graph(
                    mol=mol,
                    y_value=y_value,
                    normalize_y=self.normalize_y,
                    y_mean=y_mean,
                    y_std=y_std,
                    drug_id=drug_id,
                    task_name=task_name,
                    smiles=smiles,
                    store_fragments=self.store_fragments
                )
                graphs.append(data)
            except Exception as e:
                logger.warning(f"Failed to create graph for {drug_id}: {e}")
                n_failed += 1
        
        logger.info(f"Created {len(graphs)} graphs, {n_failed} failed")
        
        return graphs
    
    def save_graphs(
        self,
        graphs: List[Data],
        output_dir: Path,
        prefix: str = "graph"
    ) -> List[Path]:
        """
        Save graphs to individual .pt files.
        
        Args:
            graphs: List of PyG Data objects
            output_dir: Directory to save graphs
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i, g in enumerate(graphs):
            path = output_dir / f"{prefix}_{i}.pt"
            torch.save(g, path)
            paths.append(path)
        
        logger.info(f"Saved {len(graphs)} graphs to {output_dir}")
        return paths


def save_graphs(
    graphs: List[Data],
    output_dir: Path,
    prefix: str = "graph"
) -> List[Path]:
    """
    Convenience function to save graphs.
    
    Args:
        graphs: List of PyG Data objects
        output_dir: Directory to save graphs
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, g in enumerate(graphs):
        path = output_dir / f"{prefix}_{i}.pt"
        torch.save(g, path)
        paths.append(path)
    
    return paths


def load_graphs(graph_dir: Path, pattern: str = "*.pt") -> List[Data]:
    """
    Load graphs from directory.
    
    Args:
        graph_dir: Directory containing .pt files
        pattern: Glob pattern for files
        
    Returns:
        List of PyG Data objects
    """
    graph_dir = Path(graph_dir)
    graphs = []
    
    for path in sorted(graph_dir.glob(pattern)):
        try:
            g = torch.load(path, weights_only=False)
            graphs.append(g)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    
    logger.info(f"Loaded {len(graphs)} graphs from {graph_dir}")
    return graphs
