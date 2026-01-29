"""
BRICS-based molecular fragmentation for SEAL framework.

This module implements the fragment decomposition strategy used in the
SEAL (Substructure-Explainable Active Learning) framework, with extensions
for handling edge cases in kinase inhibitor chemical space.
"""

import logging
from typing import List, Tuple, Set, Optional

from .constants import HALOGEN_ATOMIC_NUMS

logger = logging.getLogger(__name__)


def brics_decompose(mol) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Perform BRICS decomposition with additional handling for:
    - Ring-nonring bond breaks
    - Inter-ring bond breaks  
    - Halogen breaks
    - High-degree atoms (>3 neighbors)
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Tuple of (cliques, breaks) where:
            cliques: List of atom index lists for each fragment
            breaks: List of [atom1, atom2] pairs for broken bonds
    """
    from rdkit.Chem import BRICS
    
    n_atoms = mol.GetNumAtoms()
    
    if n_atoms == 1:
        return [[0]], []
    
    # Initialize cliques from bonds
    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])
    
    breaks = []
    
    # Apply BRICS bond breaks
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    for bond in brics_bonds:
        bond_atoms = [bond[0][0], bond[0][1]]
        reverse = [bond[0][1], bond[0][0]]
        
        if bond_atoms in cliques:
            cliques.remove(bond_atoms)
            breaks.append(bond_atoms)
        elif reverse in cliques:
            cliques.remove(reverse)
            breaks.append(reverse)
        
        cliques.append([bond[0][0]])
        cliques.append([bond[0][1]])
    
    # Build ring membership info
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    atom_to_rings = {i: set() for i in range(n_atoms)}
    for ring_id, ring in enumerate(atom_rings):
        for atom in ring:
            atom_to_rings[atom].add(ring_id)
    
    # Break ring-nonring and inter-ring bonds
    cliques_to_remove = []
    for c in list(cliques):
        if len(c) > 1:
            atom0, atom1 = c[0], c[1]
            atom0_in_ring = mol.GetAtomWithIdx(atom0).IsInRing()
            atom1_in_ring = mol.GetAtomWithIdx(atom1).IsInRing()
            
            if atom0_in_ring and not atom1_in_ring:
                cliques_to_remove.append(c)
                cliques.append([atom1])
                breaks.append(c)
            elif atom1_in_ring and not atom0_in_ring:
                cliques_to_remove.append(c)
                cliques.append([atom0])
                breaks.append(c)
            elif (atom0_in_ring and atom1_in_ring and 
                  not (atom_to_rings[atom0] & atom_to_rings[atom1])):
                cliques_to_remove.append(c)
                cliques.append([atom0])
                cliques.append([atom1])
                breaks.append(c)
    
    for c in cliques_to_remove:
        if c in cliques:
            cliques.remove(c)
    
    # Break halogen bonds
    halogen_to_remove = []
    for c in list(cliques):
        if len(c) == 2:
            atom0, atom1 = c[0], c[1]
            an0 = mol.GetAtomWithIdx(atom0).GetAtomicNum()
            an1 = mol.GetAtomWithIdx(atom1).GetAtomicNum()
            
            if an0 in HALOGEN_ATOMIC_NUMS or an1 in HALOGEN_ATOMIC_NUMS:
                halogen_to_remove.append(c)
                cliques.append([atom0])
                cliques.append([atom1])
                breaks.append(c)
    
    for c in halogen_to_remove:
        if c in cliques:
            cliques.remove(c)
    
    # Break bonds around high-degree atoms (>3 neighbors, not in ring)
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 3 and not atom.IsInRing():
            atom_idx = atom.GetIdx()
            cliques.append([atom_idx])
            
            for nei in atom.GetNeighbors():
                nei_idx = nei.GetIdx()
                bond_forward = [nei_idx, atom_idx]
                bond_reverse = [atom_idx, nei_idx]
                
                if bond_forward in cliques:
                    cliques.remove(bond_forward)
                    breaks.append(bond_forward)
                elif bond_reverse in cliques:
                    cliques.remove(bond_reverse)
                    breaks.append(bond_reverse)
                
                cliques.append([nei_idx])
    
    # Merge overlapping cliques
    cliques = _merge_overlapping_cliques(cliques)
    
    return cliques, breaks


def _merge_overlapping_cliques(cliques: List[List[int]]) -> List[List[int]]:
    """Merge cliques that share atoms."""
    cliques = [c for c in cliques if len(c) > 0]
    
    changed = True
    while changed:
        changed = False
        new_cliques = []
        used = set()
        
        for i, c1 in enumerate(cliques):
            if i in used:
                continue
            
            merged = set(c1)
            for j, c2 in enumerate(cliques):
                if j <= i or j in used:
                    continue
                if merged & set(c2):
                    merged |= set(c2)
                    used.add(j)
                    changed = True
            
            new_cliques.append(list(merged))
            used.add(i)
        
        cliques = new_cliques
    
    return [c for c in cliques if len(c) > 0]


def get_fragment_membership_matrix(
    n_atoms: int,
    cliques: List[List[int]]
) -> List[List[float]]:
    """
    Create fragment membership matrix S.
    
    Args:
        n_atoms: Number of atoms in molecule
        cliques: List of atom index lists for each fragment
        
    Returns:
        Matrix S where S[i][j] = 1.0 if atom i belongs to fragment j
    """
    n_fragments = len(cliques)
    S = [[0.0] * n_fragments for _ in range(n_atoms)]
    
    for frag_idx, frag_atoms in enumerate(cliques):
        for atom_idx in frag_atoms:
            if 0 <= atom_idx < n_atoms:
                S[atom_idx][frag_idx] = 1.0
    
    return S


def get_edge_break_mask(
    edge_index: List[Tuple[int, int]],
    breaks: List[List[int]]
) -> List[bool]:
    """
    Create mask for edges that cross fragment boundaries.
    
    Args:
        edge_index: List of (src, dst) edge tuples
        breaks: List of broken bond pairs from decomposition
        
    Returns:
        Boolean mask where True = edge is within fragment
    """
    break_set = set()
    for b in breaks:
        break_set.add((b[0], b[1]))
        break_set.add((b[1], b[0]))
    
    mask = []
    for src, dst in edge_index:
        is_within_fragment = (src, dst) not in break_set
        mask.append(is_within_fragment)
    
    return mask


class FragmentExtractor:
    """
    Extract fragment SMILES and metadata from molecular decomposition.
    
    Provides methods to convert atom index clusters back to interpretable
    molecular fragments for SEAL-based interpretability analysis.
    """
    
    @staticmethod
    def extract_fragment_smiles(mol, atom_indices: List[int]) -> Optional[str]:
        """
        Extract canonical SMILES for a fragment defined by atom indices.
        
        Args:
            mol: RDKit Mol object
            atom_indices: List of atom indices defining the fragment
            
        Returns:
            Canonical SMILES string or None if extraction fails
        """
        from rdkit import Chem
        
        if not atom_indices or mol is None:
            return None
        
        atom_indices = sorted(atom_indices)
        atom_set = set(atom_indices)
        
        try:
            # Method 1: PathToSubmol for connected fragments
            bonds_within = []
            for bond in mol.GetBonds():
                if (bond.GetBeginAtomIdx() in atom_set and 
                    bond.GetEndAtomIdx() in atom_set):
                    bonds_within.append(bond.GetIdx())
            
            if bonds_within:
                frag = Chem.PathToSubmol(mol, bonds_within)
                if frag is not None and frag.GetNumAtoms() > 0:
                    try:
                        Chem.SanitizeMol(frag, catchErrors=True)
                        smiles = Chem.MolToSmiles(frag, canonical=True)
                        if smiles:
                            return smiles
                    except Exception:
                        pass
            
            # Method 2: Single atom fragments
            if len(atom_indices) == 1:
                atom = mol.GetAtomWithIdx(atom_indices[0])
                symbol = atom.GetSymbol()
                formal_charge = atom.GetFormalCharge()
                
                if formal_charge != 0:
                    return f"[{symbol}{formal_charge:+d}]"
                if atom.GetIsAromatic():
                    return f"[{symbol.lower()}]"
                return symbol
            
            # Method 3: Build new molecule from atoms
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
                if smiles:
                    return smiles
            except Exception:
                pass
            
            # Fallback: descriptive string
            symbols = []
            for idx in atom_indices:
                atom = mol.GetAtomWithIdx(idx)
                symbols.append(f"{atom.GetSymbol()}{atom.GetDegree()}")
            return "_".join(sorted(symbols))
            
        except Exception:
            try:
                symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in atom_indices]
                return "_".join(sorted(symbols))
            except Exception:
                return f"FRAGMENT_{len(atom_indices)}_ATOMS"
    
    @staticmethod
    def extract_all_fragments(mol, cliques: List[List[int]]) -> List[dict]:
        """
        Extract SMILES and metadata for all fragments.
        
        Args:
            mol: RDKit Mol object
            cliques: List of atom index lists for each fragment
            
        Returns:
            List of dicts with 'smiles', 'atom_indices', 'size' keys
        """
        fragments = []
        
        for frag_idx, atom_indices in enumerate(cliques):
            smiles = FragmentExtractor.extract_fragment_smiles(mol, atom_indices)
            fragments.append({
                'fragment_idx': frag_idx,
                'smiles': smiles,
                'atom_indices': atom_indices,
                'size': len(atom_indices)
            })
        
        return fragments
