"""
Explanation extraction for SEAL models.

Extracts fragment-level attributions from trained models, enabling
interpretability analysis of molecular property predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def extract_local_edges(edge_index: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract edges for a subgraph defined by node mask.
    
    Args:
        edge_index: Full graph edge index [2, E]
        node_mask: Boolean mask for nodes in subgraph
        
    Returns:
        Remapped edge index for subgraph
    """
    node_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
    idx_map = -torch.ones((node_mask.size(0),), dtype=torch.long)
    idx_map[node_idx] = torch.arange(node_idx.size(0), dtype=torch.long)
    
    edge_index_cpu = edge_index.detach().cpu()
    edge_mask = node_mask[edge_index_cpu[0]] & node_mask[edge_index_cpu[1]]
    sub_edges = edge_index_cpu[:, edge_mask]
    sub_edges = idx_map[sub_edges]
    
    return sub_edges


def extract_explanations(
    model: torch.nn.Module,
    task_name: str,
    graphs: List[Data],
    metadata_df: Optional[pd.DataFrame] = None,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract fragment-level explanations from a trained model.
    
    For each molecule, extracts:
    - Fragment contributions (sum to prediction)
    - Node importance (derived from fragment membership)
    - Fragment metadata (SMILES, atom lists)
    
    Args:
        model: Trained SEAL model
        task_name: Task to extract explanations for
        graphs: List of PyG Data objects
        metadata_df: Optional DataFrame with molecule metadata
        batch_size: Batch size for inference
        device: Device to run on
        save_path: Optional path to save explanations (.pt file)
        
    Returns:
        List of explanation dicts, one per molecule
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    explanations = []
    idx_counter = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            try:
                out = model(batch, task_name)
            except Exception as e:
                logger.warning(f"Model forward error: {e}")
                continue
            
            # Get fragment contributions
            fragment_contribs = out.get('x_cluster_transformed', out.get('fragment_contributions'))
            if fragment_contribs is None:
                raise KeyError("Model must return 'x_cluster_transformed' or 'fragment_contributions'")
            
            contribs_cpu = fragment_contribs.detach().cpu()
            
            # Get membership matrix
            s_all = batch.s.detach().cpu()
            batch_vec = batch.batch.detach().cpu()
            num_graphs = int(batch_vec.max().item()) + 1
            
            # Get predictions
            predictions = out['output'].detach().cpu().view(-1)
            
            # Get fragment info if available
            all_fragment_smiles = getattr(batch, 'fragment_smiles', None)
            all_fragment_atom_lists = getattr(batch, 'fragment_atom_lists', None)
            all_fragment_sizes = getattr(batch, 'fragment_sizes', None)
            
            node_offset = 0
            for g_idx in range(num_graphs):
                node_mask = (batch_vec == g_idx)
                num_nodes = int(node_mask.sum().item())
                
                # Extract local graph structure
                local_edge_index = extract_local_edges(
                    batch.edge_index.detach().cpu(), 
                    node_mask.detach().cpu()
                )
                local_x = batch.x[node_mask].detach().cpu().numpy()
                
                # Get target value
                local_y = None
                if hasattr(batch, 'y'):
                    try:
                        by = batch.y.detach().cpu()
                        if by.numel() == num_graphs:
                            local_y = float(by.view(-1)[g_idx].item())
                        elif node_mask.sum() > 0:
                            local_y = float(by[node_mask].view(-1)[0].item())
                    except Exception:
                        pass
                
                local_pred = float(predictions[g_idx].item())
                
                # Get fragment contributions for this graph
                s_nodes = s_all[node_offset:node_offset + num_nodes]
                
                if contribs_cpu.dim() == 3:
                    per_graph_contribs = contribs_cpu[g_idx].squeeze(-1).view(-1).numpy()
                else:
                    per_graph_contribs = contribs_cpu[g_idx].view(-1).numpy()
                
                # Compute node importance from fragment membership
                node_importance = np.zeros(num_nodes, dtype=float)
                cluster_atom_lists = []
                cluster_sizes = []
                
                if s_nodes is not None and s_nodes.numel() > 0:
                    # Hard assignment: each node belongs to argmax fragment
                    argmax_idx = s_nodes.argmax(dim=1).numpy()
                    
                    # Node importance = contribution of its fragment
                    for node_idx, frag_idx in enumerate(argmax_idx):
                        if 0 <= frag_idx < len(per_graph_contribs):
                            node_importance[node_idx] = per_graph_contribs[frag_idx]
                    
                    # Build cluster atom lists
                    n_fragments = s_nodes.shape[1]
                    for frag_idx in range(n_fragments):
                        atom_list = np.where(argmax_idx == frag_idx)[0].tolist()
                        cluster_atom_lists.append(atom_list)
                        cluster_sizes.append(len(atom_list))
                
                # Get metadata
                meta = None
                if metadata_df is not None and idx_counter < len(metadata_df):
                    meta = metadata_df.iloc[idx_counter].to_dict()
                
                # Build explanation dict
                explanation = {
                    'index': idx_counter,
                    'task_name': task_name,
                    'meta': meta,
                    'y': local_y,
                    'pred': local_pred,
                    'node_importance': node_importance,
                    'cluster_contribs': per_graph_contribs,
                    'cluster_sizes': cluster_sizes,
                    'cluster_atom_lists': cluster_atom_lists,
                    's_matrix': s_nodes.numpy() if s_nodes is not None else None,
                    'edge_index': local_edge_index.numpy(),
                    'x': local_x,
                }
                
                # Add fragment SMILES if available
                if isinstance(all_fragment_smiles, list) and len(all_fragment_smiles) > g_idx:
                    if isinstance(all_fragment_smiles[g_idx], list):
                        explanation['fragment_smiles'] = all_fragment_smiles[g_idx]
                
                if isinstance(all_fragment_atom_lists, list) and len(all_fragment_atom_lists) > g_idx:
                    if isinstance(all_fragment_atom_lists[g_idx], list):
                        explanation['fragment_atom_lists'] = all_fragment_atom_lists[g_idx]
                
                if isinstance(all_fragment_sizes, list) and len(all_fragment_sizes) > g_idx:
                    if isinstance(all_fragment_sizes[g_idx], list):
                        explanation['fragment_sizes'] = all_fragment_sizes[g_idx]
                
                explanations.append(explanation)
                idx_counter += 1
                node_offset += num_nodes
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'explanations': explanations,
            'task_name': task_name
        }, str(save_path))
        logger.info(f"Saved {len(explanations)} explanations to {save_path}")
    
    logger.info(f"Extracted {len(explanations)} explanations for {task_name}")
    
    return explanations


def extract_explanations_all_tasks(
    model: torch.nn.Module,
    task_datasets: Dict[str, Dict[str, List[Data]]],
    task_dataframes: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    output_dir: str = "./explanations",
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    splits: List[str] = None,
) -> Dict[tuple, List[Dict]]:
    """
    Extract explanations for all tasks and splits.
    
    Args:
        model: Trained SEAL model
        task_datasets: Dict of task -> split -> graphs
        task_dataframes: Optional metadata DataFrames
        output_dir: Directory to save explanation files
        batch_size: Batch size for inference
        device: Device to run on
        splits: Which splits to process (default: ['test'])
        
    Returns:
        Dict mapping (task_name, split) -> list of explanations
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if splits is None:
        splits = ['test']
    
    output_dir = Path(output_dir)
    all_explanations = {}
    
    logger.info("Extracting explanations for all tasks...")
    
    for task_name in task_datasets.keys():
        logger.info(f"\nTask: {task_name}")
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        for split in splits:
            graphs = task_datasets[task_name].get(split, [])
            if len(graphs) == 0:
                logger.info(f"  Skipping {split} (empty)")
                continue
            
            # Get metadata if available
            df = None
            if task_dataframes is not None and task_name in task_dataframes:
                df = task_dataframes[task_name].get(split)
            
            explanations = extract_explanations(
                model=model,
                task_name=task_name,
                graphs=graphs,
                metadata_df=df,
                batch_size=batch_size,
                device=device,
                save_path=str(task_dir / f"explanations_{split}.pt")
            )
            
            all_explanations[(task_name, split)] = explanations
    
    logger.info("\nExplanation extraction complete!")
    
    return all_explanations


def load_explanations(path: str) -> List[Dict[str, Any]]:
    """
    Load explanations from .pt file.
    
    Args:
        path: Path to saved explanations file
        
    Returns:
        List of explanation dicts
    """
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data.get('explanations', [])


def explanations_to_dataframe(explanations: List[Dict]) -> pd.DataFrame:
    """
    Convert explanations to a summary DataFrame.
    
    Args:
        explanations: List of explanation dicts
        
    Returns:
        DataFrame with one row per molecule
    """
    records = []
    
    for expl in explanations:
        record = {
            'index': expl.get('index'),
            'task_name': expl.get('task_name'),
            'y_true': expl.get('y'),
            'y_pred': expl.get('pred'),
            'n_fragments': len(expl.get('cluster_contribs', [])),
            'max_contrib': float(np.max(expl.get('cluster_contribs', [0]))) if len(expl.get('cluster_contribs', [])) > 0 else 0,
            'min_contrib': float(np.min(expl.get('cluster_contribs', [0]))) if len(expl.get('cluster_contribs', [])) > 0 else 0,
        }
        
        # Add metadata fields
        meta = expl.get('meta', {})
        if meta:
            for key in ['Drug_ID', 'canonical_smiles', 'smiles']:
                if key in meta:
                    record[key] = meta[key]
        
        records.append(record)
    
    return pd.DataFrame(records)


def get_top_fragments(
    explanation: Dict,
    top_k: int = 5,
    by: str = 'contribution'
) -> List[Dict]:
    """
    Get top contributing fragments from an explanation.
    
    Args:
        explanation: Single explanation dict
        top_k: Number of top fragments to return
        by: 'contribution' (absolute value) or 'positive'/'negative'
        
    Returns:
        List of fragment info dicts sorted by importance
    """
    contribs = explanation.get('cluster_contribs', [])
    atom_lists = explanation.get('cluster_atom_lists', [])
    frag_smiles = explanation.get('fragment_smiles', [])
    
    if len(contribs) == 0:
        return []
    
    fragments = []
    for i, contrib in enumerate(contribs):
        frag = {
            'fragment_idx': i,
            'contribution': float(contrib),
            'abs_contribution': abs(float(contrib)),
            'atoms': atom_lists[i] if i < len(atom_lists) else [],
            'smiles': frag_smiles[i] if i < len(frag_smiles) else None,
            'size': len(atom_lists[i]) if i < len(atom_lists) else 0,
        }
        fragments.append(frag)
    
    # Sort by requested metric
    if by == 'positive':
        fragments = [f for f in fragments if f['contribution'] > 0]
        fragments.sort(key=lambda x: x['contribution'], reverse=True)
    elif by == 'negative':
        fragments = [f for f in fragments if f['contribution'] < 0]
        fragments.sort(key=lambda x: x['contribution'])
    else:  # by absolute contribution
        fragments.sort(key=lambda x: x['abs_contribution'], reverse=True)
    
    return fragments[:top_k]
