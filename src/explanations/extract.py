"""
Explanation extraction from SEAL models.

This module provides functions for extracting fragment-level
explanations from trained SEAL models, enabling interpretability
of molecular property predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def _extract_local_edges(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor
) -> torch.Tensor:
    """Extract edges within a subgraph defined by node mask."""
    node_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
    idx_map = -torch.ones(node_mask.size(0), dtype=torch.long)
    idx_map[node_idx] = torch.arange(node_idx.size(0), dtype=torch.long)
    
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    sub_edges = edge_index[:, edge_mask]
    sub_edges = idx_map[sub_edges]
    
    return sub_edges


class MoleculeExplanation:
    """
    Container for a single molecule's explanation.
    
    Attributes:
        index: Index in the dataset
        task_name: Task for which explanation was computed
        y: Ground truth label (if available)
        pred: Model prediction
        node_importance: Per-node importance scores
        cluster_contributions: Per-fragment contribution values
        cluster_sizes: Number of atoms in each fragment
        cluster_atom_lists: Atom indices in each fragment
        smiles: Canonical SMILES (if available)
        additivity_ok: Whether contributions sum to prediction
    """
    
    def __init__(
        self,
        index: int,
        task_name: str,
        pred: float,
        node_importance: np.ndarray,
        cluster_contributions: np.ndarray,
        cluster_sizes: List[int],
        cluster_atom_lists: List[List[int]],
        y: Optional[float] = None,
        smiles: Optional[str] = None,
        drug_id: Optional[str] = None,
        s_matrix: Optional[np.ndarray] = None,
        edge_index: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        node_to_atom_map: Optional[List[int]] = None,
        additivity_ok: Optional[bool] = None,
        additivity_diff: Optional[float] = None,
        fragment_smiles: Optional[List[str]] = None,
        fragment_fingerprints: Optional[List[int]] = None,
        meta: Optional[Dict] = None
    ):
        self.index = index
        self.task_name = task_name
        self.y = y
        self.pred = pred
        self.node_importance = node_importance
        self.node_importance_raw = node_importance.copy()
        self.cluster_contributions = cluster_contributions
        self.cluster_sizes = cluster_sizes
        self.cluster_atom_lists = cluster_atom_lists
        self.smiles = smiles
        self.drug_id = drug_id
        self.s_matrix = s_matrix
        self.edge_index = edge_index
        self.x = x
        self.node_to_atom_map = node_to_atom_map
        self.additivity_ok = additivity_ok
        self.additivity_diff = additivity_diff
        self.fragment_smiles = fragment_smiles or []
        self.fragment_fingerprints = fragment_fingerprints or []
        self.meta = meta or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'index': self.index,
            'task_name': self.task_name,
            'y': self.y,
            'pred': self.pred,
            'node_importance': self.node_importance,
            'node_importance_raw': self.node_importance_raw,
            'cluster_contribs': self.cluster_contributions,
            'cluster_sizes': self.cluster_sizes,
            'cluster_atom_lists': self.cluster_atom_lists,
            'smiles': self.smiles,
            'drug_id': self.drug_id,
            's_matrix': self.s_matrix,
            'edge_index': self.edge_index,
            'x': self.x,
            'node_to_atom_map': self.node_to_atom_map,
            'additivity_ok': self.additivity_ok,
            'additivity_diff': self.additivity_diff,
            'fragment_smiles': self.fragment_smiles,
            'fragment_fingerprints': self.fragment_fingerprints,
            'meta': self.meta
        }
    
    @property
    def n_fragments(self) -> int:
        return len(self.cluster_sizes)
    
    @property
    def n_atoms(self) -> int:
        return sum(self.cluster_sizes)
    
    def get_top_fragments(self, k: int = 5, by_absolute: bool = True) -> List[int]:
        """Get indices of top-k most important fragments."""
        contribs = self.cluster_contributions
        if by_absolute:
            contribs = np.abs(contribs)
        return np.argsort(contribs)[::-1][:k].tolist()


def extract_explanations(
    model: torch.nn.Module,
    task_name: str,
    graphs: Union[List[Data], DataLoader],
    metadata_df: Optional[pd.DataFrame] = None,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    save_path: Optional[Union[str, Path]] = None,
    additivity_tol: float = 1e-4
) -> List[MoleculeExplanation]:
    """
    Extract explanations for a set of molecules.
    
    Args:
        model: Trained SEAL model
        task_name: Task to generate explanations for
        graphs: List of PyG Data objects or DataLoader
        metadata_df: Optional DataFrame with molecule metadata
        batch_size: Batch size for inference
        device: Torch device
        save_path: Optional path to save explanations
        additivity_tol: Tolerance for additivity check
        
    Returns:
        List of MoleculeExplanation objects
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    if isinstance(graphs, DataLoader):
        loader = graphs
    else:
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    explanations = []
    idx_counter = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            try:
                out = model(batch, task_name)
            except Exception as e:
                logger.error(f"Model forward error: {e}")
                continue
            
            contributions = out.get('x_cluster_transformed', out.get('fragment_contributions'))
            if contributions is None:
                raise KeyError("Model must return fragment contributions")
            
            contributions = contributions.detach().cpu()
            outputs = out['output'].detach().cpu().view(-1)
            
            s_all = batch.s.detach().cpu()
            batch_vec = batch.batch.detach().cpu()
            num_graphs = int(batch_vec.max().item()) + 1
            
            node_offset = 0
            for g in range(num_graphs):
                node_mask = (batch_vec == g)
                num_nodes = int(node_mask.sum().item())
                
                local_edge_index = _extract_local_edges(
                    batch.edge_index.cpu(), node_mask
                ).numpy()
                
                local_x = None
                if hasattr(batch, 'x'):
                    local_x = batch.x[node_mask].cpu().numpy()
                
                local_y = None
                if hasattr(batch, 'y'):
                    try:
                        by = batch.y.cpu()
                        if by.numel() == num_graphs:
                            local_y = float(by.view(-1)[g].item())
                        elif node_mask.sum() > 0:
                            local_y = float(by[node_mask].view(-1)[0].item())
                    except Exception:
                        pass
                
                local_pred = float(outputs[g].item())
                
                s_nodes = s_all[node_offset:node_offset + num_nodes]
                
                if contributions.dim() == 3:
                    per_graph_c = contributions[g].squeeze(-1).numpy()
                else:
                    per_graph_c = contributions[g].numpy()
                
                per_graph_c = np.atleast_1d(per_graph_c.flatten())
                
                node_importance = np.zeros(num_nodes, dtype=float)
                cluster_sizes = []
                cluster_atom_lists = []
                
                if s_nodes.numel() > 0 and len(per_graph_c) > 0:
                    argmax_idx = s_nodes.argmax(dim=1).numpy()
                    node_importance = np.array([
                        per_graph_c[i] if 0 <= i < len(per_graph_c) else 0.0
                        for i in argmax_idx
                    ], dtype=float)
                    
                    n_clusters = s_nodes.shape[1]
                    for j in range(n_clusters):
                        atoms = np.where(argmax_idx == j)[0].tolist()
                        cluster_atom_lists.append(atoms)
                        cluster_sizes.append(len(atoms))
                
                node_to_atom_map = list(range(num_nodes))
                
                additivity_ok = None
                additivity_diff = None
                try:
                    sum_clusters = float(np.sum(per_graph_c))
                    additivity_diff = abs(sum_clusters - local_pred)
                    additivity_ok = additivity_diff <= additivity_tol
                except Exception:
                    pass
                
                meta = None
                smiles = None
                drug_id = None
                if metadata_df is not None and idx_counter < len(metadata_df):
                    meta = metadata_df.iloc[idx_counter].to_dict()
                    smiles = meta.get('canonical_smiles') or meta.get('Canonical_Smiles')
                    drug_id = meta.get('Drug_ID') or meta.get('drug_id')
                
                fragment_smiles = []
                fragment_fps = []
                if hasattr(batch, 'fragment_smiles'):
                    fs = batch.fragment_smiles
                    if isinstance(fs, list) and len(fs) > g and isinstance(fs[g], list):
                        fragment_smiles = fs[g]
                if hasattr(batch, 'fragment_fingerprints'):
                    ff = batch.fragment_fingerprints
                    if isinstance(ff, list) and len(ff) > g:
                        fragment_fps = ff[g]
                
                expl = MoleculeExplanation(
                    index=idx_counter,
                    task_name=task_name,
                    pred=local_pred,
                    node_importance=node_importance,
                    cluster_contributions=per_graph_c,
                    cluster_sizes=cluster_sizes,
                    cluster_atom_lists=cluster_atom_lists,
                    y=local_y,
                    smiles=smiles,
                    drug_id=drug_id,
                    s_matrix=s_nodes.numpy() if s_nodes is not None else None,
                    edge_index=local_edge_index,
                    x=local_x,
                    node_to_atom_map=node_to_atom_map,
                    additivity_ok=additivity_ok,
                    additivity_diff=additivity_diff,
                    fragment_smiles=fragment_smiles,
                    fragment_fingerprints=fragment_fps,
                    meta=meta
                )
                explanations.append(expl)
                idx_counter += 1
                node_offset += num_nodes
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'explanations': [e.to_dict() for e in explanations],
            'task_name': task_name
        }, save_path)
        logger.info(f"Saved {len(explanations)} explanations to {save_path}")
    
    logger.info(f"Extracted {len(explanations)} explanations for {task_name}")
    return explanations


def load_explanations(
    path: Union[str, Path]
) -> List[MoleculeExplanation]:
    """Load explanations from a saved file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    explanations = []
    
    for d in data.get('explanations', []):
        expl = MoleculeExplanation(
            index=d['index'],
            task_name=d['task_name'],
            pred=d['pred'],
            node_importance=d['node_importance'],
            cluster_contributions=d['cluster_contribs'],
            cluster_sizes=d['cluster_sizes'],
            cluster_atom_lists=d['cluster_atom_lists'],
            y=d.get('y'),
            smiles=d.get('smiles'),
            drug_id=d.get('drug_id'),
            s_matrix=d.get('s_matrix'),
            edge_index=d.get('edge_index'),
            x=d.get('x'),
            node_to_atom_map=d.get('node_to_atom_map'),
            additivity_ok=d.get('additivity_ok'),
            additivity_diff=d.get('additivity_diff'),
            fragment_smiles=d.get('fragment_smiles', []),
            fragment_fingerprints=d.get('fragment_fingerprints', []),
            meta=d.get('meta', {})
        )
        explanations.append(expl)
    
    return explanations


def extract_all_task_explanations(
    model: torch.nn.Module,
    task_datasets: Dict[str, Any],
    task_dataframes: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Union[str, Path],
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    splits: List[str] = None
) -> Dict[tuple, List[MoleculeExplanation]]:
    """
    Extract explanations for all tasks and specified splits.
    
    Args:
        model: Trained SEAL model
        task_datasets: Dictionary of task datasets
        task_dataframes: Dictionary of metadata DataFrames per task/split
        output_dir: Base output directory
        batch_size: Batch size for inference
        device: Torch device
        splits: List of splits to process (default: ['test'])
        
    Returns:
        Dictionary mapping (task_name, split) to explanations
    """
    if splits is None:
        splits = ['test']
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(output_dir)
    all_explanations = {}
    
    logger.info("Extracting explanations for all tasks")
    
    for task_name in task_datasets.keys():
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        for split in splits:
            graphs = task_datasets[task_name].get(split, [])
            if len(graphs) == 0:
                logger.info(f"Skipping {task_name}/{split} (empty)")
                continue
            
            df = task_dataframes.get(task_name, {}).get(split)
            
            explanations = extract_explanations(
                model=model,
                task_name=task_name,
                graphs=graphs,
                metadata_df=df,
                batch_size=batch_size,
                device=device,
                save_path=task_dir / f"explanations_{split}.pt",
                additivity_tol=1e-4
            )
            
            all_explanations[(task_name, split)] = explanations
    
    return all_explanations
