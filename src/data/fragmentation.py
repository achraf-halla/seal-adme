"""
Molecular fragmentation for SEAL-ADME.

This module provides BRICS-based molecular decomposition for creating
fragment-aware graph representations of molecules.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Fragment extraction utilities
# =============================================================================

class FragmentExtractor:
    """Helper class to extract and store fragment information from molecules."""
    
    @staticmethod
    def extract_fragment_smiles(mol, atom_indices: List[int]) -> Optional[str]:
        """
        Extract canonical SMILES for a fragment defined by atom indices.
        
        Uses multiple fallback methods to handle various fragment types:
        1. PathToSubmol for connected fragments
        2. Direct atom extraction for single atoms
        3. Manual molecule building for complex fragments
        4. Descriptive string as fallback
        
        Args:
            mol: RDKit molecule object
            atom_indices: List of atom indices defining the fragment
            
        Returns:
            Canonical SMILES string for the fragment, or descriptive fallback
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit not installed")
        
        if not atom_indices or mol is None:
            return None
        
        atom_indices = sorted(atom_indices)
        atom_set = set(atom_indices)
        
        try:
            # Method 1: Use PathToSubmol for connected fragments
            bonds_within = []
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx in atom_set and end_idx in atom_set:
                    bonds_within.append(bond.GetIdx())
            
            if bonds_within:
                frag = Chem.PathToSubmol(mol, bonds_within)
                if frag is not None and frag.GetNumAtoms() > 0:
                    try:
                        Chem.SanitizeMol(frag, catchErrors=True)
                        smiles = Chem.MolToSmiles(frag, canonical=True)
                        if smiles and smiles != '':
                            return smiles
                    except Exception:
                        pass
            
            # Method 2: Handle single atom fragments
            if len(atom_indices) == 1:
                atom = mol.GetAtomWithIdx(atom_indices[0])
                symbol = atom.GetSymbol()
                formal_charge = atom.GetFormalCharge()
                
                if formal_charge != 0:
                    return f"[{symbol}{formal_charge:+d}]"
                if atom.GetIsAromatic():
                    return f"[{symbol.lower()}]"
                return symbol
            
            # Method 3: Build new molecule from atoms manually
            new_mol = Chem.RWMol()
            old_to_new = {}
            
            for old_idx in atom_indices:
                atom = mol.GetAtomWithIdx(old_idx)
                new_idx = new_mol.AddAtom(atom)
                old_to_new[old_idx] = new_idx
            
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx in atom_set and end_idx in atom_set:
                    new_mol.AddBond(
                        old_to_new[begin_idx],
                        old_to_new[end_idx],
                        bond.GetBondType()
                    )
            
            frag = new_mol.GetMol()
            try:
                Chem.SanitizeMol(frag, catchErrors=True)
                smiles = Chem.MolToSmiles(frag, canonical=True)
                if smiles and smiles != '':
                    return smiles
            except Exception:
                pass
            
            # Method 4: Fallback to descriptive string
            symbols = []
            for idx in atom_indices:
                atom = mol.GetAtomWithIdx(idx)
                symbols.append(f"{atom.GetSymbol()}{atom.GetDegree()}")
            return "_".join(sorted(symbols))
        
        except Exception:
            # Ultimate fallback
            try:
                symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in atom_indices]
                return "_".join(sorted(symbols))
            except Exception:
                return f"FRAGMENT_{len(atom_indices)}_ATOMS"
    
    @staticmethod
    def compute_fragment_fingerprint(smiles: str, radius: int = 2) -> Optional[str]:
        """
        Compute Morgan fingerprint hash for a fragment.
        
        Args:
            smiles: Fragment SMILES
            radius: Morgan fingerprint radius
            
        Returns:
            String hash of the fingerprint, or None if invalid
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit not installed")
        
        if smiles is None:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprint(mol, radius=radius)
            # Create a hashable representation
            return str(hash(tuple(sorted(fp.GetNonzeroElements().items()))))
        except Exception:
            return None


# =============================================================================
# BRICS decomposition
# =============================================================================

def brics_decomp(mol) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Perform BRICS decomposition on a molecule.
    
    BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures)
    breaks molecules at strategic bonds to yield chemically meaningful fragments.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of:
        - cliques: List of atom index lists, each defining a fragment
        - breaks: List of (atom1, atom2) tuples representing broken bonds
    """
    try:
        from rdkit.Chem import BRICS
    except ImportError:
        raise ImportError("RDKit not installed")
    
    if mol is None:
        return None, None
    
    try:
        # Get BRICS bonds (bonds to break)
        brics_bonds = BRICS.FindBRICSBonds(mol)
        
        if not brics_bonds:
            # No BRICS bonds found - entire molecule is one fragment
            return [[i for i in range(mol.GetNumAtoms())]], []
        
        # Extract the atom pairs from BRICS bonds
        breaks = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
        
        # Find connected components after removing BRICS bonds
        num_atoms = mol.GetNumAtoms()
        
        # Build adjacency without broken bonds
        broken_set = set()
        for a1, a2 in breaks:
            broken_set.add((min(a1, a2), max(a1, a2)))
        
        # Union-Find for connected components
        parent = list(range(num_atoms))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union atoms connected by non-broken bonds
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            pair = (min(a1, a2), max(a1, a2))
            if pair not in broken_set:
                union(a1, a2)
        
        # Group atoms by their component
        from collections import defaultdict
        components = defaultdict(list)
        for i in range(num_atoms):
            components[find(i)].append(i)
        
        cliques = list(components.values())
        
        return cliques, breaks
    
    except Exception as e:
        logger.warning(f"BRICS decomposition failed: {e}")
        return None, None


def brics_decomp_extra(mol) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Enhanced BRICS decomposition with additional handling.
    
    Falls back to treating each atom as its own fragment if BRICS fails
    or produces empty results.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (cliques, breaks)
    """
    cliques, breaks = brics_decomp(mol)
    
    if cliques is None or len(cliques) == 0:
        # Fallback: each atom is its own fragment
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            cliques = [[i] for i in range(num_atoms)]
            breaks = []
        else:
            cliques = []
            breaks = []
    
    return cliques, breaks


# =============================================================================
# Fragment metadata extraction
# =============================================================================

def extract_fragment_metadata(
    mol,
    cliques: List[List[int]],
    compute_fingerprints: bool = True,
    fp_radius: int = 2,
) -> Dict:
    """
    Extract comprehensive metadata for all fragments in a molecule.
    
    Args:
        mol: RDKit molecule object
        cliques: List of atom index lists defining fragments
        compute_fingerprints: Whether to compute Morgan fingerprints
        fp_radius: Radius for Morgan fingerprints
        
    Returns:
        Dictionary containing:
        - fragment_smiles: List of SMILES for each fragment
        - fragment_atom_lists: List of atom indices for each fragment
        - fragment_sizes: List of fragment sizes (number of atoms)
        - fragment_fingerprints: List of fingerprint hashes (optional)
    """
    extractor = FragmentExtractor()
    
    fragment_smiles = []
    fragment_atom_lists = []
    fragment_sizes = []
    fragment_fingerprints = []
    
    for clique in cliques:
        # Extract SMILES
        smiles = extractor.extract_fragment_smiles(mol, clique)
        fragment_smiles.append(smiles)
        
        # Store atom indices
        fragment_atom_lists.append(clique)
        
        # Fragment size
        fragment_sizes.append(len(clique))
        
        # Fingerprint (optional)
        if compute_fingerprints:
            fp = extractor.compute_fragment_fingerprint(smiles, radius=fp_radius)
            fragment_fingerprints.append(fp)
    
    result = {
        "fragment_smiles": fragment_smiles,
        "fragment_atom_lists": fragment_atom_lists,
        "fragment_sizes": fragment_sizes,
    }
    
    if compute_fingerprints:
        result["fragment_fingerprints"] = fragment_fingerprints
    
    return result


# =============================================================================
# Cluster assignment matrices
# =============================================================================

def create_cluster_assignment_matrix(
    cliques: List[List[int]],
    num_atoms: int,
) -> np.ndarray:
    """
    Create a soft cluster assignment matrix S.
    
    S[i, j] = 1 if atom i belongs to cluster/fragment j, else 0.
    
    Args:
        cliques: List of atom index lists
        num_atoms: Total number of atoms in molecule
        
    Returns:
        Assignment matrix of shape (num_atoms, num_clusters)
    """
    num_clusters = len(cliques)
    s = np.zeros((num_atoms, num_clusters), dtype=np.float32)
    
    for cluster_idx, clique in enumerate(cliques):
        for atom_idx in clique:
            if atom_idx < num_atoms:
                s[atom_idx, cluster_idx] = 1.0
    
    return s


def create_atom_wise_assignment_matrix(
    cliques: List[List[int]],
    num_atoms: int,
) -> np.ndarray:
    """
    Create atom-wise cluster assignment matrix S_node.
    
    This is an alternative representation where each row sums to 1,
    representing the fraction of membership in each cluster.
    
    Args:
        cliques: List of atom index lists
        num_atoms: Total number of atoms in molecule
        
    Returns:
        Normalized assignment matrix of shape (num_atoms, num_clusters)
    """
    s = create_cluster_assignment_matrix(cliques, num_atoms)
    
    # Normalize rows to sum to 1
    row_sums = s.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    s_node = s / row_sums
    
    return s_node


# =============================================================================
# Edge masking for inter-fragment edges
# =============================================================================

def mask_broken_edges(
    edge_index: np.ndarray,
    breaks: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Create a mask indicating which edges cross fragment boundaries.
    
    Args:
        edge_index: Edge index array of shape (2, num_edges)
        breaks: List of (atom1, atom2) tuples for broken bonds
        
    Returns:
        Mask array of shape (num_edges,) where 1 = inter-fragment edge
    """
    if edge_index.size == 0:
        return np.array([], dtype=np.float32)
    
    num_edges = edge_index.shape[1]
    mask = np.zeros(num_edges, dtype=np.float32)
    
    # Create set of broken bond pairs (both directions)
    broken_set = set()
    for a1, a2 in breaks:
        broken_set.add((a1, a2))
        broken_set.add((a2, a1))
    
    # Mark inter-fragment edges
    for i in range(num_edges):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        if (src, dst) in broken_set:
            mask[i] = 1.0
    
    return mask


def identify_inter_fragment_edges(
    edge_index: np.ndarray,
    cliques: List[List[int]],
) -> np.ndarray:
    """
    Identify edges that connect different fragments.
    
    Args:
        edge_index: Edge index array of shape (2, num_edges)
        cliques: List of atom index lists defining fragments
        
    Returns:
        Mask array where 1 = inter-fragment edge, 0 = intra-fragment edge
    """
    if edge_index.size == 0:
        return np.array([], dtype=np.float32)
    
    num_edges = edge_index.shape[1]
    
    # Create atom-to-fragment mapping
    atom_to_fragment = {}
    for frag_idx, clique in enumerate(cliques):
        for atom_idx in clique:
            atom_to_fragment[atom_idx] = frag_idx
    
    mask = np.zeros(num_edges, dtype=np.float32)
    
    for i in range(num_edges):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        
        src_frag = atom_to_fragment.get(src, -1)
        dst_frag = atom_to_fragment.get(dst, -1)
        
        if src_frag != dst_frag:
            mask[i] = 1.0
    
    return mask
