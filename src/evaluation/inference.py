"""
Inference utilities for SEAL models.

Provides functions for:
- Batch prediction on new molecules
- Explanation extraction during inference
- Results saving and loading
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def predict_task(
    model: torch.nn.Module,
    task_name: str,
    graphs: List[Data],
    batch_size: int = 128,
    device: torch.device = None,
    return_contributions: bool = False
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Make predictions for a single task.
    
    Args:
        model: Trained SEAL model
        task_name: Name of the task to predict
        graphs: List of molecular graphs
        batch_size: Batch size for inference
        device: Torch device
        return_contributions: Whether to return fragment contributions
        
    Returns:
        Predictions array, or dict with predictions and contributions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    if len(graphs) == 0:
        if return_contributions:
            return {'predictions': np.array([]), 'contributions': []}
        return np.array([])
    
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    predictions = []
    contributions = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, task_name)
            
            preds = out['output'].view(-1).cpu().numpy()
            predictions.append(preds)
            
            if return_contributions:
                contribs = out.get('fragment_contributions')
                if contribs is not None:
                    contribs = contribs.cpu().numpy()
                    # Split by batch
                    batch_vec = batch.batch.cpu().numpy()
                    for g_idx in range(int(batch_vec.max()) + 1):
                        contributions.append(contribs[g_idx])
    
    predictions = np.concatenate(predictions)
    
    if return_contributions:
        return {
            'predictions': predictions,
            'contributions': contributions
        }
    
    return predictions


def predict_all_tasks(
    model: torch.nn.Module,
    graphs: List[Data],
    batch_size: int = 128,
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    """
    Make predictions for all tasks in the model.
    
    Args:
        model: Trained SEAL model
        graphs: List of molecular graphs
        batch_size: Batch size
        device: Torch device
        
    Returns:
        Dict mapping task name to predictions
    """
    predictions = {}
    
    for task_name in model.task_names:
        preds = predict_task(model, task_name, graphs, batch_size, device)
        predictions[task_name] = preds
        logger.info(f"  {task_name}: {len(preds)} predictions")
    
    return predictions


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dict with spearman, pearson, rmse
    """
    sp, _ = spearmanr(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'spearman': float(sp) if not np.isnan(sp) else -1.0,
        'pearson': float(pr) if not np.isnan(pr) else -1.0,
        'rmse': float(rmse),
        'n_samples': len(y_true)
    }


def extract_drug_ids(graphs: List[Data]) -> List[str]:
    """Extract Drug_ID from graph attributes."""
    drug_ids = []
    
    for i, g in enumerate(graphs):
        drug_id = None
        
        # Try different attribute names
        for attr in ['drug_id', 'Drug_ID']:
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
    output_path: Path,
    y_true: np.ndarray = None,
    additional_columns: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Save predictions to CSV with rankings.
    
    Args:
        predictions: Prediction array
        drug_ids: Drug identifiers
        task_name: Task name
        output_path: Output file path
        y_true: Optional ground truth values
        additional_columns: Optional additional data columns
        
    Returns:
        Saved DataFrame
    """
    data = {
        'Drug_ID': drug_ids,
        f'{task_name}_pred': predictions
    }
    
    if y_true is not None:
        data[f'{task_name}_true'] = y_true
        data[f'{task_name}_residual'] = y_true - predictions
    
    if additional_columns:
        data.update(additional_columns)
    
    df = pd.DataFrame(data)
    
    # Add rank (higher prediction = better rank for potency tasks)
    df['rank'] = df[f'{task_name}_pred'].rank(method='min', ascending=False).astype(int)
    df = df.sort_values('rank')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved predictions to {output_path}")
    return df


class InferenceRunner:
    """
    Run inference and save results for a trained model.
    
    Args:
        model: Trained SEAL model
        output_dir: Directory for results
        device: Torch device
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        device: torch.device = None
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        self.model = self.model.to(self.device)
    
    def run_task(
        self,
        task_name: str,
        graphs: List[Data],
        dataset_name: str,
        extract_explanations: bool = True,
        batch_size: int = 128
    ) -> Dict[str, Any]:
        """
        Run inference for a single task on a dataset.
        
        Args:
            task_name: Task name
            graphs: List of molecular graphs
            dataset_name: Name for the dataset (used in output filenames)
            extract_explanations: Whether to extract explanations
            batch_size: Batch size
            
        Returns:
            Results dictionary
        """
        logger.info(f"Running inference: {task_name} on {dataset_name}")
        
        if len(graphs) == 0:
            logger.warning("No graphs provided")
            return {}
        
        task_dir = self.output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions
        result = predict_task(
            self.model,
            task_name,
            graphs,
            batch_size=batch_size,
            device=self.device,
            return_contributions=extract_explanations
        )
        
        if extract_explanations:
            predictions = result['predictions']
            contributions = result['contributions']
        else:
            predictions = result
            contributions = None
        
        # Get drug IDs
        drug_ids = extract_drug_ids(graphs)
        
        # Get ground truth if available
        y_true = None
        try:
            y_true = np.array([g.y.item() for g in graphs])
        except:
            pass
        
        # Save predictions CSV
        csv_path = task_dir / f"predictions_{dataset_name}.csv"
        df = save_predictions_csv(
            predictions,
            drug_ids,
            task_name,
            csv_path,
            y_true=y_true
        )
        
        results = {
            'predictions': predictions,
            'drug_ids': drug_ids,
            'dataframe': df
        }
        
        # Compute metrics if ground truth available
        if y_true is not None:
            metrics = evaluate_predictions(y_true, predictions)
            results['metrics'] = metrics
            
            # Save metrics
            metrics_path = task_dir / f"metrics_{dataset_name}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(
                f"  Spearman: {metrics['spearman']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}"
            )
        
        # Extract and save explanations
        if extract_explanations and contributions:
            from .explanations import extract_explanations, save_explanations
            
            # Build metadata DataFrame
            meta_list = []
            for g, drug_id in zip(graphs, drug_ids):
                meta = {'Drug_ID': drug_id}
                if hasattr(g, 'canonical_smiles'):
                    meta['canonical_smiles'] = g.canonical_smiles
                meta_list.append(meta)
            
            meta_df = pd.DataFrame(meta_list)
            
            explanations = extract_explanations(
                self.model,
                task_name,
                graphs,
                metadata_df=meta_df,
                batch_size=8,
                device=self.device
            )
            
            save_explanations(
                explanations,
                task_dir / f"explanations_{dataset_name}.pt",
                task_name
            )
            
            results['explanations'] = explanations
        
        return results
    
    def run_all_tasks(
        self,
        graphs: List[Data],
        dataset_name: str,
        extract_explanations: bool = True,
        batch_size: int = 128
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run inference for all tasks.
        
        Args:
            graphs: List of molecular graphs
            dataset_name: Dataset name
            extract_explanations: Whether to extract explanations
            batch_size: Batch size
            
        Returns:
            Nested results dict {task_name: results}
        """
        logger.info(f"Running inference on {dataset_name} ({len(graphs)} molecules)")
        
        all_results = {}
        
        for task_name in self.model.task_names:
            results = self.run_task(
                task_name,
                graphs,
                dataset_name,
                extract_explanations=extract_explanations,
                batch_size=batch_size
            )
            all_results[task_name] = results
        
        # Save combined predictions
        combined = pd.DataFrame({'Drug_ID': extract_drug_ids(graphs)})
        for task_name, results in all_results.items():
            if 'predictions' in results:
                combined[f'{task_name}_pred'] = results['predictions']
        
        combined.to_csv(
            self.output_dir / f"all_predictions_{dataset_name}.csv",
            index=False
        )
        
        logger.info(f"Results saved to {self.output_dir}")
        return all_results


def load_graphs_from_directory(
    graph_dir: Path,
    pattern: str = "*.pt"
) -> List[Data]:
    """
    Load graphs from a directory.
    
    Args:
        graph_dir: Directory containing .pt files
        pattern: Glob pattern for files
        
    Returns:
        List of Data objects
    """
    graphs = []
    
    for fpath in sorted(Path(graph_dir).glob(pattern)):
        try:
            g = torch.load(fpath, weights_only=False)
            graphs.append(g)
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")
    
    logger.info(f"Loaded {len(graphs)} graphs from {graph_dir}")
    return graphs
