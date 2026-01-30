"""
Graph featurization for SEAL-ADME using BRICS decomposition.

Creates PyG Data objects with:
- Atom features (25 features per atom)
- Edge connectivity from bonds
- Fragment membership matrix (s) from BRICS decomposition
- Edge mask for intra/inter-fragment message passing
- Fragment metadata for interpretability
"""

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Feature vocabularies
DEFAULT_ATOM_TYPE_SET = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "Other"]

PAULING_ENERGY = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "B": 2.04
}
DEFAULT_EN = 2.50

# Halogen atomic numbers for BRICS decomposition
HALOGEN_ATOMIC_NUMS = {9, 17, 35, 53, 85, 117}


class FragmentExtractor:
    """Helper for extracting fragment info from molecules."""

    @staticmethod
    def extract_fragment_smiles(mol: Chem.Mol, atom_indices: List[int]) -> Optional[str]:
        """Extract canonical SMILES for fragment defined by atom indices."""
        if not atom_indices or mol is None:
            return None

        atom_indices = sorted(atom_indices)
        atom_set = set(atom_indices)

        try:
            # Try PathToSubmol for connected fragments
            bonds_within = []
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
                    bonds_within.append(bond.GetIdx())

            if bonds_within:
                frag = Chem.PathToSubmol(mol, bonds_within)
                if frag is not None and frag.GetNumAtoms() > 0:
                    try:
                        Chem.SanitizeMol(frag, catchErrors=True)
                        smiles = Chem.MolToSmiles(frag, canonical=True)
                        if smiles and smiles != '':
                            return smiles
                    except:
                        pass

            # Single atom fragments
            if len(atom_indices) == 1:
                atom = mol.GetAtomWithIdx(atom_indices[0])
                symbol = atom.GetSymbol()
                formal_charge = atom.GetFormalCharge()
                if formal_charge != 0:
                    return f"[{symbol}{formal_charge:+d}]"
                if atom.GetIsAromatic():
                    return f"[{symbol.lower()}]"
                return symbol

            # Build new molecule from atoms
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
            except:
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
            except:
                return f"FRAGMENT_{len(atom_indices)}_ATOMS"

    @staticmethod
    def compute_fragment_fingerprint(smiles: str, radius: int = 2) -> Optional[str]:
        """Compute Morgan fingerprint for fragment."""
        if smiles is None:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprint(mol, radius=radius)
            return str(hash(tuple(sorted(fp.GetNonzeroElements().items()))))
        except:
            return None


class GraphFeaturizer:
    """
    Graph featurizer that creates PyG Data objects with fragment info.
    
    Features per atom (25 total):
    - Atom type one-hot (11)
    - Degree (1)
    - Formal charge (1)
    - Is aromatic (1)
    - Hybridization one-hot (4)
    - Valence (1)
    - Is in ring (1)
    - Num Hs (1)
    - Atomic num scaled (1)
    - Atom mass scaled (1)
    - Electronegativity scaled (1)
    - Gasteiger charge (1)
    """

    def __init__(
        self,
        y_column: str = 'Y',
        smiles_col: str = 'Drug',
        include_optional: bool = True,
        include_numeric: bool = True,
        include_gasteiger: bool = True,
        store_fragments: bool = True
    ):
        self.y_column = y_column
        self.smiles_col = smiles_col
        self.include_optional = include_optional
        self.include_numeric = include_numeric
        self.include_gasteiger = include_gasteiger
        self.store_fragments = store_fragments
        self.fragment_extractor = FragmentExtractor()

    @staticmethod
    def one_of_k_encoding_unk(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))

    def _gasteiger_charges(self, mol):
        if not self.include_gasteiger:
            return np.zeros((mol.GetNumAtoms(),), dtype=float)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = []
            for a in mol.GetAtoms():
                c = a.GetProp('_GasteigerCharge') if a.HasProp('_GasteigerCharge') else "0.0"
                try:
                    charges.append(float(c))
                except Exception:
                    charges.append(0.0)
            return np.array(charges, dtype=float)
        except Exception:
            return np.zeros((mol.GetNumAtoms(),), dtype=float)

    def get_atom_features(self, atom, gasteiger_charge=None):
        feats = []
        # Atom type (11)
        feats.extend(self.one_of_k_encoding_unk(atom.GetSymbol(), DEFAULT_ATOM_TYPE_SET))
        
        if self.include_optional:
            # Degree (1)
            feats.append(float(atom.GetDegree()))
            # Formal charge (1)
            feats.append(float(atom.GetFormalCharge()))
            # Is aromatic (1)
            feats.append(1.0 if atom.GetIsAromatic() else 0.0)
            # Hybridization (4)
            hyb = str(atom.GetHybridization()).replace('.', '').upper()
            feats.extend(self.one_of_k_encoding_unk(hyb, DEFAULT_HYBRIDIZATION_SET))
            # Valence (1)
            feats.append(float(atom.GetImplicitValence()))
            # Is in ring (1)
            feats.append(1.0 if atom.IsInRing() else 0.0)
            # Num Hs (1)
            feats.append(float(atom.GetTotalNumHs()))
        
        if self.include_numeric:
            # Atomic num scaled (1)
            feats.append(float(atom.GetAtomicNum()) / 100.0)
            # Atom mass scaled (1)
            feats.append(float(atom.GetMass()) / 200.0)
            # Electronegativity scaled (1)
            feats.append(float(PAULING_ENERGY.get(atom.GetSymbol(), DEFAULT_EN)) / 4.0)
            # Gasteiger charge (1)
            feats.append(float(gasteiger_charge) if gasteiger_charge is not None else 0.0)
        
        return feats

    def brics_decomp_extra(self, mol) -> Tuple[List[List[int]], List[List[int]]]:
        """BRICS decomposition with additional handling for edge cases."""
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        breaks = []

        # Start with bonds as cliques
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            cliques.append([a1, a2])

        # Find BRICS bonds
        res = list(BRICS.FindBRICSBonds(mol))
        for bond in res:
            bond_atoms = [bond[0][0], bond[0][1]]
            if bond_atoms in cliques:
                cliques.remove(bond_atoms)
                breaks.append(bond_atoms)
            else:
                reverse_bond = [bond[0][1], bond[0][0]]
                if reverse_bond in cliques:
                    cliques.remove(reverse_bond)
                breaks.append(reverse_bond)
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

        # Ring handling
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        atom_to_rings = {i: set() for i in range(mol.GetNumAtoms())}
        for id_ring, ring in enumerate(atom_rings):
            for atom in ring:
                atom_to_rings[atom].add(id_ring)

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
                elif (atom0_in_ring and atom1_in_ring) and not (atom_to_rings[atom0] & atom_to_rings[atom1]):
                    cliques_to_remove.append(c)
                    cliques.append([atom0])
                    cliques.append([atom1])
                    breaks.append(c)

        for c in cliques_to_remove:
            if c in cliques:
                cliques.remove(c)

        # Handle halogen bonds
        halogen_bonds_to_remove = []
        for c in list(cliques):
            if len(c) == 2:
                atom0, atom1 = c[0], c[1]
                if mol.GetAtomWithIdx(atom0).GetAtomicNum() in HALOGEN_ATOMIC_NUMS or \
                   mol.GetAtomWithIdx(atom1).GetAtomicNum() in HALOGEN_ATOMIC_NUMS:
                    halogen_bonds_to_remove.append(c)
                    cliques.append([atom0])
                    cliques.append([atom1])
                    breaks.append(c)

        for c in halogen_bonds_to_remove:
            if c in cliques:
                cliques.remove(c)

        # Atoms with degree > 3 outside rings
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
        for c in range(len(cliques) - 1):
            if c >= len(cliques):
                break
            for k in range(c + 1, len(cliques)):
                if k >= len(cliques):
                    break
                if len(set(cliques[c]) & set(cliques[k])) > 0:
                    cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                    cliques[k] = []
            cliques = [c for c in cliques if len(c) > 0]

        cliques = [c for c in cliques if len(c) > 0]
        return cliques, breaks

    def create_s(self, cliques, num_atoms):
        """Create fragment membership matrix S."""
        s = torch.zeros((num_atoms, len(cliques)), dtype=torch.float32)
        for clique_idx, clique in enumerate(cliques):
            for atom_idx in clique:
                s[atom_idx, clique_idx] = 1.0
        return s

    def mask_broken_edges(self, edge_index, breaks):
        """Create mask for edges (1 = within fragment, 0 = between fragments)."""
        if edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)
        num_edges = edge_index.size(1)
        mask = torch.ones(num_edges, dtype=torch.float32)
        broken_edges = set(frozenset([a, b]) for a, b in breaks)
        for i in range(num_edges):
            edge = frozenset([edge_index[0, i].item(), edge_index[1, i].item()])
            if edge in broken_edges:
                mask[i] = 0.0
        return mask

    def extract_fragment_metadata(self, mol: Chem.Mol, cliques: List[List[int]]) -> Dict:
        """Extract fragment SMILES and metadata."""
        fragment_metadata = {
            'fragment_smiles': [],
            'fragment_atom_lists': [],
            'fragment_sizes': [],
        }

        for clique in cliques:
            frag_smiles = self.fragment_extractor.extract_fragment_smiles(mol, clique)
            fragment_metadata['fragment_smiles'].append(frag_smiles)
            fragment_metadata['fragment_atom_lists'].append(clique)
            fragment_metadata['fragment_sizes'].append(len(clique))

        return fragment_metadata

    def smiles_to_graph(
        self,
        smiles: str,
        y_value: float = 0.0,
        drug_id: str = "",
        task_name: str = "",
        normalize_y: bool = False,
        y_mean: float = 0.0,
        y_std: float = 1.0
    ) -> Optional[Data]:
        """Convert a single SMILES to PyG Data object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Normalize y if needed
        if normalize_y and y_std > 0:
            y_norm = (y_value - y_mean) / y_std
        else:
            y_norm = y_value

        g_charges = self._gasteiger_charges(mol)

        # Build edge index
        edges = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edges.append((start, end))
            edges.append((end, start))

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edges_arr = np.array(edges).T
            edge_index = torch.LongTensor(edges_arr)

        # Atom features
        nodes = []
        for i, atom in enumerate(mol.GetAtoms()):
            gc = g_charges[i] if g_charges.size > i else 0.0
            res = self.get_atom_features(atom, gasteiger_charge=gc)
            nodes.append(res)

        if len(nodes) > 0:
            x = torch.FloatTensor(np.array(nodes))
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            return None

        num_atoms = mol.GetNumAtoms()
        cliques, breaks = self.brics_decomp_extra(mol)
        if cliques is None or len(cliques) == 0:
            cliques = [[i] for i in range(num_atoms)]

        s = self.create_s(cliques, num_atoms)
        mask = self.mask_broken_edges(edge_index, breaks)

        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            s=s,
            y=torch.FloatTensor([float(y_norm)]),
            mask=mask
        )

        # Store metadata
        data.drug_id = drug_id
        data.task_name = task_name
        data.smiles = smiles

        # Store fragment info
        if self.store_fragments:
            fragment_meta = self.extract_fragment_metadata(mol, cliques)
            data.fragment_smiles = fragment_meta['fragment_smiles']
            data.fragment_atom_lists = fragment_meta['fragment_atom_lists']
            data.fragment_sizes = fragment_meta['fragment_sizes']

        return data

    def __call__(
        self,
        df: pd.DataFrame,
        task_name: str = "",
        normalize_y: bool = False,
        y_mean: float = 0.0,
        y_std: float = 1.0
    ) -> List[Data]:
        """
        Convert DataFrame to list of PyG Data objects.
        
        Args:
            df: DataFrame with 'Drug' (SMILES) and 'Y' columns
            task_name: Name of the task
            normalize_y: Whether to normalize Y values
            y_mean: Mean for normalization
            y_std: Std for normalization
            
        Returns:
            List of PyG Data objects
        """
        graphs = []
        n_failed = 0

        for idx, row in df.iterrows():
            try:
                y = float(row[self.y_column])
            except Exception:
                n_failed += 1
                continue

            smiles = row[self.smiles_col]
            drug_id = row.get("Drug_ID", f"mol_{idx}")

            graph = self.smiles_to_graph(
                smiles=smiles,
                y_value=y,
                drug_id=str(drug_id),
                task_name=task_name,
                normalize_y=normalize_y,
                y_mean=y_mean,
                y_std=y_std
            )

            if graph is not None:
                graphs.append(graph)
            else:
                n_failed += 1

        logger.info(f"Created {len(graphs)} graphs, {n_failed} failed for task '{task_name}'")
        return graphs


def save_graphs(graphs: List[Data], output_dir: Path, prefix: str = "graph") -> List[Path]:
    """Save graphs to individual .pt files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, g in enumerate(graphs):
        path = output_dir / f"{prefix}_{i}.pt"
        torch.save(g, path)
        paths.append(path)
    
    return paths


def load_graphs(graph_dir: Path, pattern: str = "*.pt") -> List[Data]:
    """Load graphs from directory."""
    graph_dir = Path(graph_dir)
    graphs = []
    
    for path in sorted(graph_dir.glob(pattern)):
        try:
            g = torch.load(path, weights_only=False)
            graphs.append(g)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    
    return graphs
