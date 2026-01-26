"""
Explanation extraction for SEAL models.

Extracts fragment-level attributions from trained models to provide
interpretable explanations for predictions.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def extract_explanations(
    model: torch.nn.Module,
    task_name: str,
    graphs: List[Data],
    metadata_df: Optional[pd.DataFrame] = None,
    batch_size: int = 8,
    device: torch.device = None,
    additivity_tol: float = 1e-4
) -> List[Dict[str, Any]]:
    """
    Extract fragment-level explanations from a trained model.
    
    For each molecule, extracts:
    - Per-fragment contributions to the prediction
    - Fragment SMILES and atom mappings
    - Node-level importance scores
    - Additivity verification (sum of contributions vs output)
    
    Args:
        model: Trained SEAL model
        task_name: Name of the task to explain
        graphs: List of molecular graphs
        metadata_df: Optional DataFrame with molecule metadata
        batch_size: Batch size for inference
        device: Torch device
        additivity_tol: Tolerance for additivity check
        
    Returns:
        List of explanation dictionaries, one per molecule
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
            
            # Forward pass
            out = model(batch, task_name)
            
            # Get fragment contributions
            fragment_contribs = out.get('fragment_contributions')
            if fragment_contribs is None:
                fragment_contribs = out.get('x_cluster_transformed')
            
            if fragment_contribs is None:
                logger.warning("No fragment contributions found in model output")
                continue
            
            fragment_contribs = fragment_contribs.detach().cpu()
            predictions = out['output'].detach().cpu().view(-1)
            
            # Get fragment membership
            s_matrix = batch.s.detach().cpu()
            batch_vec = batch.batch.detach().cpu()
            
            num_graphs = int(batch_vec.max().item()) + 1
            node_offset = 0
            
            for g_idx in range(num_graphs):
                # Get nodes for this graph
                node_mask = (batch_vec == g_idx)
                num_nodes = int(node_mask.sum().item())
                
                # Extract local S matrix
                local_s = s_matrix[node_offset:node_offset + num_nodes]
                
                # Get fragment contributions for this graph
                if fragment_contribs.dim() == 3:
                    local_contribs = fragment_contribs[g_idx].squeeze(-1).numpy()
                else:
                    local_contribs = fragment_contribs[g_idx].numpy()
                
                # Compute node importance via S matrix
                # Each node gets the contribution of its fragment
                node_importance = np.zeros(num_nodes)
                cluster_atom_lists = []
                
                if local_s.numel() > 0 and len(local_contribs) > 0:
                    # Assign each node to its fragment (argmax of S)
                    fragment_assignment = local_s.argmax(dim=1).numpy()
                    
                    for node_idx, frag_idx in enumerate(fragment_assignment):
                        if 0 <= frag_idx < len(local_contribs):
                            node_importance[node_idx] = local_contribs[frag_idx]
                    
                    # Build cluster atom lists
                    n_frags = local_s.shape[1]
                    for frag_idx in range(n_frags):
                        atoms = np.where(fragment_assignment == frag_idx)[0].tolist()
                        cluster_atom_lists.append(atoms)
                
                # Get ground truth and prediction
                y_true = None
                if hasattr(batch, 'y'):
                    y_batch = batch.y.detach().cpu()
                    if y_batch.numel() == num_graphs:
                        y_true = float(y_batch[g_idx].item())
                
                y_pred = float(predictions[g_idx].item())
                
                # Check additivity
                contrib_sum = float(np.sum(local_contribs))
                additivity_diff = abs(contrib_sum - y_pred)
                additivity_ok = additivity_diff <= additivity_tol
                
                # Get metadata
                meta = None
                if metadata_df is not None and idx_counter < len(metadata_df):
                    meta = metadata_df.iloc[idx_counter].to_dict()
                
                # Get fragment info from graph attributes
                fragment_smiles = getattr(graphs[idx_counter], 'fragment_smiles', None)
                fragment_atom_lists = getattr(graphs[idx_counter], 'fragment_atom_lists', None)
                fragment_sizes = getattr(graphs[idx_counter], 'fragment_sizes', None)
                
                explanation = {
                    'index': idx_counter,
                    'task_name': task_name,
                    'meta': meta,
                    'y': y_true,
                    'pred': y_pred,
                    'node_importance': node_importance,
                    'cluster_contribs': local_contribs,
                    'cluster_atom_lists': cluster_atom_lists,
                    's_matrix': local_s.numpy(),
                    'additivity_ok': additivity_ok,
                    'additivity_diff': additivity_diff,
                    'fragment_smiles': fragment_smiles,
                    'fragment_atom_lists': fragment_atom_lists,
                    'fragment_sizes': fragment_sizes
                }
                
                explanations.append(explanation)
                idx_counter += 1
                node_offset += num_nodes
    
    logger.info(f"Extracted {len(explanations)} explanations for {task_name}")
    
    return explanations


def save_explanations(
    explanations: List[Dict],
    output_path: Path,
    task_name: str
):
    """Save explanations to a .pt file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'explanations': explanations,
        'task_name': task_name
    }, output_path)
    
    logger.info(f"Saved {len(explanations)} explanations to {output_path}")


def load_explanations(path: Path) -> tuple:
    """Load explanations from a .pt file."""
    data = torch.load(path, map_location='cpu')
    return data.get('explanations', []), data.get('task_name', '')


def aggregate_fragment_importance(
    explanations: List[Dict],
    fragment_key: str = 'fragment_smiles'
) -> pd.DataFrame:
    """
    Aggregate fragment importance across all molecules.
    
    Args:
        explanations: List of explanation dictionaries
        fragment_key: Key for fragment identifiers
        
    Returns:
        DataFrame with fragment importance statistics
    """
    fragment_stats = {}
    
    for expl in explanations:
        frags = expl.get(fragment_key, [])
        contribs = expl.get('cluster_contribs', [])
        
        if frags is None or contribs is None:
            continue
        
        for frag, contrib in zip(frags, contribs):
            if frag is None:
                continue
            
            if frag not in fragment_stats:
                fragment_stats[frag] = {
                    'contributions': [],
                    'count': 0
                }
            
            fragment_stats[frag]['contributions'].append(contrib)
            fragment_stats[frag]['count'] += 1
    
    # Compute statistics
    rows = []
    for frag, stats in fragment_stats.items():
        contribs = np.array(stats['contributions'])
        rows.append({
            'fragment': frag,
            'count': stats['count'],
            'mean_contribution': float(np.mean(contribs)),
            'std_contribution': float(np.std(contribs)),
            'min_contribution': float(np.min(contribs)),
            'max_contribution': float(np.max(contribs))
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('mean_contribution', ascending=False)
    
    return df


def get_top_fragments(
    explanations: List[Dict],
    n_positive: int = 10,
    n_negative: int = 10
) -> Dict[str, List[Dict]]:
    """
    Get fragments with highest positive and negative contributions.
    
    Args:
        explanations: List of explanation dictionaries
        n_positive: Number of top positive fragments to return
        n_negative: Number of top negative fragments to return
        
    Returns:
        Dictionary with 'positive' and 'negative' fragment lists
    """
    df = aggregate_fragment_importance(explanations)
    
    return {
        'positive': df.head(n_positive).to_dict('records'),
        'negative': df.tail(n_negative).to_dict('records')
    }
