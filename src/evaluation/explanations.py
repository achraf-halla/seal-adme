"""
Fragment-level explanation extraction for SEAL models.

Extracts per-fragment contributions that sum to the final prediction,
enabling interpretable molecular property predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr

# Use our custom dataloader with padding
from ..training.datasets import create_dataloader, collate_with_padding

logger = logging.getLogger(__name__)


def _extract_local_edges(edge_index, node_mask):
    """Extract edges for a single graph from a batch."""
    node_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
    idx_map = -torch.ones((node_mask.size(0),), dtype=torch.long)
    idx_map[node_idx] = torch.arange(node_idx.size(0), dtype=torch.long)
    edge_index_cpu = edge_index.detach().cpu()
    edge_mask = node_mask[edge_index_cpu[0]] & node_mask[edge_index_cpu[1]]
    sub_e = edge_index_cpu[:, edge_mask]
    sub_e = idx_map[sub_e]
    return sub_e


def spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation, handling edge cases."""
    if len(y_true) < 2:
        return float('nan')
    r = spearmanr(y_true, y_pred).correlation
    return float('nan') if np.isnan(r) else float(r)


def extract_explanations_for_task(
    model,
    task_name: str,
    graphs: List,
    dataset_df: pd.DataFrame = None,
    batch_size: int = 8,
    device: str = None,
    save_path: str = None,
    additivity_tol: float = 1e-4,
) -> List[Dict]:
    """
    Extract fragment-level explanations for a task.
    
    Args:
        model: Trained MultiTaskRegressionModel
        task_name: Name of the task
        graphs: List of PyG Data objects
        dataset_df: Optional DataFrame with metadata
        batch_size: Batch size for inference
        device: Device to use
        save_path: Path to save explanations
        additivity_tol: Tolerance for additivity check
        
    Returns:
        List of explanation dictionaries
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Use our custom dataloader with padding
    loader = create_dataloader(graphs, batch_size=batch_size, shuffle=False)
    
    explanations = []
    idx_counter = 0
    
    node_to_atom_attrs = ("node_to_atom_map", "node_idx_to_atom_idx", "node_to_atom")
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            try:
                out = model(batch, task_name)
            except Exception as e:
                logger.warning(f"Model forward error: {e}")
                continue
            
            # Get fragment contributions
            xt = out.get("x_cluster_transformed", None)
            if xt is None:
                raise KeyError("Model must return 'x_cluster_transformed' (fragment contributions)")
            
            xt_cpu = xt.detach().cpu()
            
            # Get fragment membership matrix
            s_all = getattr(batch, "s", None)
            if s_all is None:
                raise KeyError("Batch must contain 's' (fragment membership matrix)")
            s_all_cpu = s_all.detach().cpu()
            
            batch_vec = batch.batch.detach().cpu()
            num_graphs = int(batch_vec.max().item()) + 1
            
            # Get predictions
            raw_outputs = out.get("output", out.get("y_pred", None))
            if raw_outputs is None:
                raise KeyError("Model did not return 'output'")
            raw_outputs = raw_outputs.detach().cpu().view(-1)
            
            # Get fragment metadata if available
            all_fragment_smiles = getattr(batch, "fragment_smiles", None)
            all_fragment_atom_lists = getattr(batch, "fragment_atom_lists", None)
            all_fragment_sizes = getattr(batch, "fragment_sizes", None)
            
            # Get node to atom mapping
            batch_node_to_atom_map = None
            for attr in node_to_atom_attrs:
                maybe = getattr(batch, attr, None)
                if maybe is not None:
                    batch_node_to_atom_map = maybe
                    break
            
            node_offset = 0
            for g in range(num_graphs):
                node_mask = (batch_vec == g)
                num_nodes = int(node_mask.sum().item())
                
                # Extract local graph structure
                local_edge_index = _extract_local_edges(
                    batch.edge_index.detach().cpu(), 
                    node_mask.detach().cpu()
                )
                local_x = batch.x[node_mask].detach().cpu().numpy() if hasattr(batch, "x") else None
                
                # Get ground truth
                local_y = None
                if hasattr(batch, "y"):
                    try:
                        by = batch.y.detach().cpu()
                        if by.numel() == num_graphs:
                            local_y = float(by.view(-1)[g].item())
                        else:
                            local_y = float(by[node_mask].view(-1)[0].item()) if node_mask.sum() > 0 else None
                    except Exception:
                        local_y = None
                
                local_pred = float(raw_outputs[g].item())
                
                # Get fragment membership for this graph
                s_nodes = s_all_cpu[node_offset: node_offset + num_nodes]
                
                # Get fragment contributions for this graph
                if xt_cpu.dim() == 3:
                    per_graph_c = xt_cpu[g].squeeze(-1).view(-1).numpy()
                elif xt_cpu.dim() == 2:
                    per_graph_c = xt_cpu[g].view(-1).numpy()
                else:
                    raise ValueError(f"Unsupported x_cluster_transformed shape: {tuple(xt_cpu.shape)}")
                
                # Compute node importance from fragment contributions
                node_importance_raw = np.zeros((num_nodes,), dtype=float)
                node_importance = np.zeros((num_nodes,), dtype=float)
                cluster_sizes = []
                cluster_atom_lists = []
                
                if s_nodes is not None and s_nodes.numel() > 0 and per_graph_c is not None:
                    argmax_idx = s_nodes.argmax(dim=1).detach().cpu().numpy()
                    node_scores = np.array([
                        per_graph_c[i] if (0 <= i < len(per_graph_c)) else 0.0 
                        for i in argmax_idx
                    ], dtype=float)
                    node_importance_raw = node_scores.copy()
                    node_importance = node_importance_raw.copy()
                    
                    K = s_nodes.shape[1]
                    for j in range(K):
                        atom_list_local = np.where(argmax_idx == j)[0].tolist()
                        cluster_atom_lists.append(atom_list_local)
                        cluster_sizes.append(len(atom_list_local))
                
                # Node to atom mapping
                node_to_atom_map = None
                if batch_node_to_atom_map is not None:
                    try:
                        if isinstance(batch_node_to_atom_map, torch.Tensor):
                            sliced = batch_node_to_atom_map.detach().cpu().numpy()[
                                node_offset: node_offset + num_nodes
                            ].tolist()
                            node_to_atom_map = [int(x) for x in sliced]
                        elif isinstance(batch_node_to_atom_map, (list, tuple)):
                            if len(batch_node_to_atom_map) == num_graphs:
                                node_to_atom_map = batch_node_to_atom_map[g]
                            else:
                                node_to_atom_map = batch_node_to_atom_map[node_offset: node_offset + num_nodes]
                    except Exception:
                        node_to_atom_map = None
                
                if node_to_atom_map is None:
                    node_to_atom_map = list(range(num_nodes))
                
                # Check additivity (fragment contributions should sum to prediction)
                additivity_ok = None
                additivity_diff = None
                try:
                    sum_clusters = float(np.sum(per_graph_c)) if per_graph_c is not None else 0.0
                    additivity_diff = abs(sum_clusters - local_pred)
                    additivity_ok = (additivity_diff <= additivity_tol)
                except Exception:
                    additivity_ok = False
                
                # Get metadata
                meta = None
                if dataset_df is not None and idx_counter < len(dataset_df):
                    meta = dataset_df.iloc[idx_counter].to_dict()
                
                # Build explanation dict
                explanation = {
                    "index": idx_counter,
                    "task_name": task_name,
                    "meta": meta,
                    "y": local_y,
                    "pred": local_pred,
                    "node_importance_raw": node_importance_raw,
                    "node_importance": node_importance,
                    "cluster_contribs": per_graph_c if per_graph_c is not None else np.array([]),
                    "cluster_sizes": cluster_sizes,
                    "cluster_atom_lists": cluster_atom_lists,
                    "s_matrix": s_nodes,
                    "edge_index": local_edge_index.numpy(),
                    "x": local_x,
                    "node_to_atom_map": node_to_atom_map,
                    "additivity_ok": additivity_ok,
                    "additivity_diff": additivity_diff,
                }
                
                # Add fragment metadata if available
                if isinstance(all_fragment_smiles, list) and len(all_fragment_smiles) > g:
                    if isinstance(all_fragment_smiles[g], list):
                        explanation['fragment_smiles'] = all_fragment_smiles[g]
                if isinstance(all_fragment_atom_lists, list) and len(all_fragment_atom_lists) > g:
                    explanation['fragment_atom_lists'] = all_fragment_atom_lists[g]
                if isinstance(all_fragment_sizes, list) and len(all_fragment_sizes) > g:
                    explanation['fragment_sizes'] = all_fragment_sizes[g]
                
                explanations.append(explanation)
                idx_counter += 1
                node_offset += num_nodes
    
    # Save explanations
    if save_path is not None:
        try:
            save_pathp = Path(save_path)
            save_pathp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"explanations": explanations, "task_name": task_name}, str(save_pathp))
            logger.info(f"Saved {len(explanations)} explanations to {save_pathp}")
        except Exception as e:
            logger.warning(f"Could not save explanations: {e}")
    
    has_fragments = any('fragment_smiles' in expl for expl in explanations)
    logger.info(f"Extracted {len(explanations)} explanations for {task_name}")
    if has_fragments:
        logger.info("Fragment information included")
    
    return explanations


def predict_task_on_graphs(
    model,
    task_name: str,
    graphs: List,
    batch_size: int = 128,
    device: str = None
) -> np.ndarray:
    """
    Make predictions on a list of graphs.
    
    Args:
        model: Trained model
        task_name: Name of the task
        graphs: List of PyG Data objects
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Array of predictions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    loader = create_dataloader(graphs, batch_size=batch_size, shuffle=False)
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, task_name)
            preds = out["output"].view(-1).detach().cpu().numpy()
            predictions.append(preds)
    
    if len(predictions) == 0:
        return np.array([])
    
    return np.concatenate(predictions)


def extract_drug_ids_from_graphs(graphs: List) -> List[str]:
    """Extract Drug IDs from graph attributes."""
    drug_ids = []
    for i, g in enumerate(graphs):
        drug_id = None
        
        # Try various attribute names
        for attr in ['drug_id', 'Drug_ID', 'mol_id']:
            if hasattr(g, attr):
                drug_id = getattr(g, attr)
                break
        
        # Try meta dict
        if drug_id is None and hasattr(g, 'meta'):
            meta = g.meta
            if isinstance(meta, dict):
                drug_id = meta.get('Drug_ID') or meta.get('drug_id')
        
        if drug_id is None:
            drug_id = f"mol_{i}"
        
        drug_ids.append(str(drug_id))
    
    return drug_ids


def save_predictions_csv(
    predictions: np.ndarray,
    drug_ids: List[str],
    task_name: str,
    output_path: str,
    y_true: np.ndarray = None,
    norm_stats: Dict = None
) -> pd.DataFrame:
    """
    Save predictions to CSV with optional denormalization.
    
    Args:
        predictions: Model predictions (normalized)
        drug_ids: List of molecule IDs
        task_name: Name of the task
        output_path: Output CSV path
        y_true: Ground truth values (normalized)
        norm_stats: Dict with 'mean' and 'std' for denormalization
        
    Returns:
        DataFrame with predictions
    """
    data = {
        'Drug_ID': drug_ids,
        f'{task_name}_pred': predictions
    }
    
    # Denormalize if stats provided
    if norm_stats is not None:
        mean, std = norm_stats.get('mean', 0.0), norm_stats.get('std', 1.0)
        data[f'{task_name}_pred_denorm'] = predictions * std + mean
    
    if y_true is not None:
        data[f'{task_name}_true'] = y_true
        data[f'{task_name}_residual'] = y_true - predictions
        
        if norm_stats is not None:
            mean, std = norm_stats.get('mean', 0.0), norm_stats.get('std', 1.0)
            data[f'{task_name}_true_denorm'] = y_true * std + mean
    
    df = pd.DataFrame(data)
    df["rank"] = df[f"{task_name}_pred"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values("rank").reset_index(drop=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    return df


def run_inference_and_save(
    model,
    task_name: str,
    graphs: List,
    dataset_name: str,
    output_dir: str,
    norm_stats: Dict = None,
    extract_explanations: bool = True,
    batch_size: int = 128,
    device: str = None
) -> Dict:
    """
    Run inference on graphs and save results.
    
    Args:
        model: Trained model
        task_name: Name of the task
        graphs: List of PyG Data objects
        dataset_name: Name for output files
        output_dir: Output directory
        norm_stats: Dict with 'mean' and 'std' for denormalization
        extract_explanations: Whether to extract explanations
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Dict with predictions, metrics, and optionally explanations
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if len(graphs) == 0:
        logger.warning(f"No graphs provided for {dataset_name}")
        return None
    
    logger.info(f"Processing {dataset_name} ({len(graphs)} molecules)")
    
    # Make predictions
    predictions = predict_task_on_graphs(
        model=model,
        task_name=task_name,
        graphs=graphs,
        batch_size=batch_size,
        device=device
    )
    
    drug_ids = extract_drug_ids_from_graphs(graphs)
    
    # Get ground truth if available
    y_true = None
    try:
        y_true = np.array([g.y.item() for g in graphs])
    except:
        pass
    
    # Save predictions
    output_path = Path(output_dir)
    csv_path = output_path / f"predictions_{dataset_name}.csv"
    df = save_predictions_csv(
        predictions=predictions,
        drug_ids=drug_ids,
        task_name=task_name,
        output_path=str(csv_path),
        y_true=y_true,
        norm_stats=norm_stats
    )
    
    results = {
        'predictions': predictions,
        'drug_ids': drug_ids,
        'y_true': y_true,
        'dataframe': df
    }
    
    # Compute metrics if ground truth available
    if y_true is not None:
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        spearman = spearman_scorer(y_true, predictions)
        pearson_r = pearsonr(y_true, predictions)[0] if len(y_true) > 1 else float('nan')
        
        # Denormalized RMSE
        rmse_denorm = rmse * norm_stats.get('std', 1.0) if norm_stats else rmse
        
        logger.info(f"Metrics: RMSE={rmse_denorm:.4f}, Spearman={spearman:.4f}, Pearson={pearson_r:.4f}")
        
        metrics = {
            'dataset': dataset_name,
            'task': task_name,
            'n_samples': len(graphs),
            'rmse': float(rmse),
            'rmse_denorm': float(rmse_denorm),
            'spearman': float(spearman),
            'pearson': float(pearson_r)
        }
        
        metrics_path = output_path / f"metrics_{dataset_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        results['metrics'] = metrics
    
    # Extract explanations
    if extract_explanations:
        logger.info("Extracting explanations...")
        
        # Build metadata DataFrame
        dataset_df = None
        try:
            meta_list = []
            for g, drug_id in zip(graphs, drug_ids):
                meta = {'Drug_ID': drug_id}
                
                if hasattr(g, 'smiles'):
                    meta['canonical_smiles'] = g.smiles
                elif hasattr(g, 'canonical_smiles'):
                    meta['canonical_smiles'] = g.canonical_smiles
                
                meta_list.append(meta)
            
            dataset_df = pd.DataFrame(meta_list)
        except:
            pass
        
        explanations = extract_explanations_for_task(
            model=model,
            task_name=task_name,
            graphs=graphs,
            dataset_df=dataset_df,
            batch_size=8,
            device=device,
            save_path=str(output_path / f"explanations_{dataset_name}.pt")
        )
        
        results['explanations'] = explanations
    
    return results


def run_inference_all_tasks(
    model,
    task_datasets: Dict[str, List],
    base_output_dir: str,
    norm_stats: Dict[str, Dict] = None,
    extract_explanations: bool = True,
    batch_size: int = 128,
    device: str = None
) -> Dict:
    """
    Run inference on all tasks and datasets.
    
    Args:
        model: Trained multi-task model
        task_datasets: Dict mapping dataset_name -> list of graphs
        base_output_dir: Base output directory
        norm_stats: Dict mapping task_name -> {'mean': float, 'std': float}
        extract_explanations: Whether to extract explanations
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Dict of (task_name, dataset_name) -> results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_dir = Path(base_output_dir)
    all_results = {}
    
    logger.info("=" * 60)
    logger.info("RUNNING INFERENCE ON ALL TASKS")
    logger.info("=" * 60)
    
    for task_name in model.task_names:
        logger.info(f"\nTask: {task_name}")
        logger.info("-" * 40)
        
        task_dir = base_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        task_norm = norm_stats.get(task_name) if norm_stats else None
        
        for dataset_name, graphs in task_datasets.items():
            results = run_inference_and_save(
                model=model,
                task_name=task_name,
                graphs=graphs,
                dataset_name=dataset_name,
                output_dir=str(task_dir),
                norm_stats=task_norm,
                extract_explanations=extract_explanations,
                batch_size=batch_size,
                device=device
            )
            
            if results is not None:
                all_results[(task_name, dataset_name)] = results
    
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 60)
    
    return all_results
