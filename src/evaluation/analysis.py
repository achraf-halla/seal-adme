"""
Analysis utilities for model comparison and drug discovery.

Provides functionality for:
- Cross-task correlation analysis
- Pareto optimization (multi-objective)
- SA Score calculation
- Model ensemble analysis
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


# =============================================================================
# TASK CONFIGURATION
# =============================================================================

TASK_LABELS = {
    'AKA': 'Aurora A',
    'AKB': 'Aurora B',
    'solubility_aqsoldb': 'Solubility',
    'caco2': 'Caco-2',
    'half_life_obach': 'Half-life'
}

TASK_UNITS = {
    'AKA': 'pIC50',
    'AKB': 'pIC50',
    'solubility_aqsoldb': 'log mol/L',
    'caco2': 'log Papp',
    'half_life_obach': 'log t½'
}

# Higher is better for these tasks
MAXIMIZE_TASKS = {'AKA', 'AKB', 'caco2'}
MINIMIZE_TASKS = {'solubility_aqsoldb', 'half_life_obach'}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions(
    base_dir: Path,
    model_name: str,
    task: str,
    dataset: str
) -> Optional[pd.DataFrame]:
    """
    Load prediction CSV from model output directory.
    
    Args:
        base_dir: Base directory for model outputs
        model_name: Model identifier (e.g., 'gcn0.5', 'gin0.0')
        task: Task name
        dataset: Dataset name (e.g., 'AKA_test', 'gen_graphs')
        
    Returns:
        DataFrame with predictions or None if not found
    """
    # Try common path patterns
    paths_to_try = [
        base_dir / model_name / 'finetune' / 'inference' / task / f'predictions_{dataset}.csv',
        base_dir / model_name / 'inference' / task / f'predictions_{dataset}.csv',
        base_dir / model_name / task / f'predictions_{dataset}.csv',
    ]
    
    for fpath in paths_to_try:
        if fpath.exists():
            df = pd.read_csv(fpath)
            return df
    
    logger.warning(f"Predictions not found for {model_name}/{task}/{dataset}")
    return None


def load_smiles_mapping(path: Path) -> Dict[str, str]:
    """Load Drug_ID to SMILES mapping from parquet file."""
    if not path.exists():
        return {}
    
    df = pd.read_parquet(path)
    smiles_col = 'canonical_smiles' if 'canonical_smiles' in df.columns else 'smiles'
    id_col = 'Drug_ID' if 'Drug_ID' in df.columns else df.columns[0]
    
    return df.set_index(id_col)[smiles_col].to_dict()


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_task_correlations(
    predictions: Dict[str, np.ndarray],
    tasks: List[str] = None
) -> pd.DataFrame:
    """
    Compute pairwise Spearman correlations between task predictions.
    
    Args:
        predictions: Dict mapping task name to prediction array
        tasks: Task names to include (default: all)
        
    Returns:
        Correlation matrix as DataFrame
    """
    if tasks is None:
        tasks = list(predictions.keys())
    
    n_tasks = len(tasks)
    corr_matrix = np.zeros((n_tasks, n_tasks))
    
    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if task1 not in predictions or task2 not in predictions:
                corr_matrix[i, j] = np.nan
            elif i == j:
                corr_matrix[i, j] = 1.0
            else:
                r, _ = spearmanr(predictions[task1], predictions[task2])
                corr_matrix[i, j] = r
    
    return pd.DataFrame(corr_matrix, index=tasks, columns=tasks)


def compare_correlations(
    corr_exp: pd.DataFrame,
    corr_gen: pd.DataFrame,
    tasks: List[str]
) -> pd.DataFrame:
    """
    Compare correlations between experimental and generated datasets.
    
    Args:
        corr_exp: Correlation matrix for experimental data
        corr_gen: Correlation matrix for generated data
        tasks: Task names
        
    Returns:
        DataFrame with correlation comparisons
    """
    rows = []
    
    for t1, t2 in combinations(tasks, 2):
        exp_r = corr_exp.loc[t1, t2]
        gen_r = corr_gen.loc[t1, t2]
        
        # Categorize change
        if abs(exp_r) < 0.1 and abs(gen_r) > 0.3:
            change = 'emerged'
        elif abs(exp_r) > 0.3 and abs(gen_r) < 0.1:
            change = 'disappeared'
        elif exp_r * gen_r < 0 and (abs(exp_r) > 0.2 or abs(gen_r) > 0.2):
            change = 'reversed'
        elif abs(gen_r) > abs(exp_r) + 0.3:
            change = 'strengthened'
        elif abs(gen_r) < abs(exp_r) - 0.3:
            change = 'weakened'
        else:
            change = 'stable'
        
        rows.append({
            'task1': t1,
            'task2': t2,
            'experimental': exp_r,
            'generated': gen_r,
            'delta': gen_r - exp_r,
            'change': change
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# SA SCORE CALCULATION
# =============================================================================

def compute_sa_score(smiles: str) -> float:
    """
    Compute Synthetic Accessibility Score for a molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        SA Score (1-10, lower is better)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RDConfig
        import os
        import sys
        
        # Import SA scorer from RDKit contrib
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        
        import sascorer
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0  # Max score for invalid
        
        return sascorer.calculateScore(mol)
    
    except Exception as e:
        logger.warning(f"SA score calculation failed: {e}")
        return np.nan


def compute_sa_scores_batch(smiles_list: List[str]) -> np.ndarray:
    """Compute SA scores for a batch of molecules."""
    return np.array([compute_sa_score(s) for s in smiles_list])


# =============================================================================
# PARETO OPTIMIZATION
# =============================================================================

def is_pareto_dominated(
    point: np.ndarray,
    other_points: np.ndarray,
    maximize: np.ndarray = None
) -> bool:
    """
    Check if a point is dominated by any other point.
    
    Args:
        point: Single point [n_objectives]
        other_points: Other points [n_points, n_objectives]
        maximize: Boolean array indicating which objectives to maximize
        
    Returns:
        True if point is dominated
    """
    if maximize is None:
        maximize = np.ones(len(point), dtype=bool)
    
    # Convert to minimization
    point_min = np.where(maximize, -point, point)
    others_min = np.where(maximize, -other_points, other_points)
    
    for other in others_min:
        # Check if 'other' dominates 'point'
        if np.all(other <= point_min) and np.any(other < point_min):
            return True
    
    return False


def find_pareto_front(
    points: np.ndarray,
    maximize: np.ndarray = None
) -> np.ndarray:
    """
    Find indices of Pareto-optimal points.
    
    Args:
        points: Array of shape [n_points, n_objectives]
        maximize: Boolean array indicating which objectives to maximize
        
    Returns:
        Boolean mask of Pareto-optimal points
    """
    n_points = len(points)
    is_optimal = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_optimal[i]:
            other_indices = np.concatenate([
                np.arange(0, i),
                np.arange(i + 1, n_points)
            ])
            other_points = points[other_indices[is_optimal[other_indices]]]
            
            if len(other_points) > 0:
                is_optimal[i] = not is_pareto_dominated(
                    points[i], other_points, maximize
                )
    
    return is_optimal


def pareto_rank(
    points: np.ndarray,
    maximize: np.ndarray = None,
    max_rank: int = 10
) -> np.ndarray:
    """
    Compute Pareto rank for each point (1 = Pareto front).
    
    Args:
        points: Array of shape [n_points, n_objectives]
        maximize: Boolean array for maximization objectives
        max_rank: Maximum rank to compute
        
    Returns:
        Rank array (1 = best, higher = worse)
    """
    n_points = len(points)
    ranks = np.zeros(n_points, dtype=int)
    remaining = np.ones(n_points, dtype=bool)
    
    for rank in range(1, max_rank + 1):
        if not remaining.any():
            break
        
        remaining_points = points[remaining]
        remaining_indices = np.where(remaining)[0]
        
        is_front = find_pareto_front(remaining_points, maximize)
        
        for i, idx in enumerate(remaining_indices):
            if is_front[i]:
                ranks[idx] = rank
                remaining[idx] = False
    
    # Assign max_rank + 1 to any remaining
    ranks[remaining] = max_rank + 1
    
    return ranks


class ParetoAnalyzer:
    """
    Multi-objective Pareto analysis for drug candidates.
    
    Args:
        objectives: List of objective names
        maximize: List of booleans (True = maximize, False = minimize)
    """
    
    def __init__(
        self,
        objectives: List[str],
        maximize: List[bool] = None
    ):
        self.objectives = objectives
        
        if maximize is None:
            # Default: maximize potency, permeability; minimize others
            maximize = [obj in MAXIMIZE_TASKS for obj in objectives]
        
        self.maximize = np.array(maximize)
    
    def analyze(
        self,
        df: pd.DataFrame,
        include_sa_score: bool = True
    ) -> pd.DataFrame:
        """
        Perform Pareto analysis on predictions.
        
        Args:
            df: DataFrame with columns for each objective
            include_sa_score: Whether to include SA score as objective
            
        Returns:
            DataFrame with Pareto ranks and fronts
        """
        result = df.copy()
        
        # Build objective matrix
        obj_cols = [c for c in self.objectives if c in df.columns]
        
        if include_sa_score and 'sa_score' in df.columns:
            obj_cols.append('sa_score')
            maximize = np.concatenate([
                self.maximize[:len(obj_cols) - 1],
                [False]  # Minimize SA score
            ])
        else:
            maximize = self.maximize[:len(obj_cols)]
        
        points = df[obj_cols].values
        
        # Handle missing values
        valid_mask = ~np.any(np.isnan(points), axis=1)
        
        ranks = np.full(len(df), -1, dtype=int)
        ranks[valid_mask] = pareto_rank(points[valid_mask], maximize)
        
        result['pareto_rank'] = ranks
        result['is_pareto_front'] = (ranks == 1)
        
        return result
    
    def get_pareto_front(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get molecules on the Pareto front."""
        analyzed = self.analyze(df)
        return analyzed[analyzed['is_pareto_front']]


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(
    predictions: Dict[str, Dict[str, np.ndarray]],
    ground_truth: Dict[str, np.ndarray] = None
) -> pd.DataFrame:
    """
    Compare multiple models across tasks.
    
    Args:
        predictions: Nested dict {model_name: {task_name: predictions}}
        ground_truth: Optional dict {task_name: true_values}
        
    Returns:
        DataFrame with model comparison metrics
    """
    rows = []
    
    models = list(predictions.keys())
    tasks = set()
    for model_preds in predictions.values():
        tasks.update(model_preds.keys())
    
    for task in tasks:
        for model in models:
            if task not in predictions[model]:
                continue
            
            preds = predictions[model][task]
            
            row = {
                'model': model,
                'task': task,
                'n_samples': len(preds),
                'mean_pred': float(np.mean(preds)),
                'std_pred': float(np.std(preds))
            }
            
            if ground_truth and task in ground_truth:
                y_true = ground_truth[task]
                if len(y_true) == len(preds):
                    r, _ = spearmanr(y_true, preds)
                    row['spearman'] = float(r) if not np.isnan(r) else None
                    
                    pr, _ = pearsonr(y_true, preds)
                    row['pearson'] = float(pr) if not np.isnan(pr) else None
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def ensemble_predictions(
    predictions: Dict[str, np.ndarray],
    weights: Dict[str, float] = None,
    method: str = 'mean'
) -> np.ndarray:
    """
    Combine predictions from multiple models.
    
    Args:
        predictions: Dict mapping model name to prediction array
        weights: Optional weights for each model
        method: 'mean', 'median', or 'weighted'
        
    Returns:
        Combined predictions
    """
    arrays = list(predictions.values())
    
    if not arrays:
        raise ValueError("No predictions provided")
    
    stacked = np.stack(arrays, axis=0)
    
    if method == 'median':
        return np.median(stacked, axis=0)
    elif method == 'weighted' and weights:
        w = np.array([weights.get(m, 1.0) for m in predictions.keys()])
        w = w / w.sum()
        return np.average(stacked, axis=0, weights=w)
    else:
        return np.mean(stacked, axis=0)
