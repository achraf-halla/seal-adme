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
    
    # Handle single atom molecules
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
            
            # Ring to non-ring bond
            if atom0_in_ring and not atom1_in_ring:
                cliques_to_remove.append(c)
                cliques.append([atom1])
                breaks.append(c)
            elif atom1_in_ring and not atom0_in_ring:
                cliques_to_remove.append(c)
                cliques.append([atom0])
                breaks.append(c)
            # Bond between different ring systems
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
            radius: Fingerprint radius
            
        Returns:
            String hash of fingerprint or None
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        if smiles is None:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprint(mol, radius=radius)
            return str(hash(tuple(sorted(fp.GetNonzeroElements().items()))))
        except Exception:
            return None
