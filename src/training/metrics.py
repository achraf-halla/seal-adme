"""
Metrics for evaluating SEAL model predictions.

This module provides metric functions for both classification
(pretraining) and regression (finetuning) tasks.
"""

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def safe_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUROC with error handling for edge cases."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return float('nan')


def safe_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUPRC with error handling for edge cases."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(average_precision_score(y_true, y_pred))
    except Exception:
        return float('nan')


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation with error handling."""
    if len(y_true) < 2:
        return float('nan')
    try:
        r, _ = spearmanr(y_true, y_pred)
        return float(r) if not np.isnan(r) else float('nan')
    except Exception:
        return float('nan')


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation with error handling."""
    if len(y_true) < 2:
        return float('nan')
    try:
        r, _ = pearsonr(y_true, y_pred)
        return float(r) if not np.isnan(r) else float('nan')
    except Exception:
        return float('nan')


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE with error handling."""
    if len(y_true) == 0:
        return float('nan')
    try:
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return float('nan')


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE with error handling."""
    if len(y_true) == 0:
        return float('nan')
    try:
        return float(mean_absolute_error(y_true, y_pred))
    except Exception:
        return float('nan')


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² with error handling."""
    if len(y_true) < 2:
        return float('nan')
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float('nan')


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for binary classification tasks.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        
    Returns:
        Dictionary with auroc and auprc
    """
    return {
        'auroc': safe_auroc(y_true, y_pred),
        'auprc': safe_auprc(y_true, y_pred),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for regression tasks.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with spearman, pearson, rmse, mae, and r2
    """
    return {
        'spearman': safe_spearman(y_true, y_pred),
        'pearson': safe_pearson(y_true, y_pred),
        'rmse': safe_rmse(y_true, y_pred),
        'mae': safe_mae(y_true, y_pred),
        'r2': safe_r2(y_true, y_pred),
    }


def aggregate_task_metrics(
    task_metrics: Dict[str, Dict[str, float]],
    metric_name: str
) -> float:
    """
    Aggregate a metric across multiple tasks.
    
    Args:
        task_metrics: Dictionary mapping task names to metric dicts
        metric_name: Name of metric to aggregate
        
    Returns:
        Mean of the metric across tasks (ignoring NaN values)
    """
    values = [
        metrics[metric_name]
        for metrics in task_metrics.values()
        if not np.isnan(metrics.get(metric_name, float('nan')))
    ]
    return float(np.mean(values)) if values else float('nan')


class MetricTracker:
    """
    Tracks metrics over training epochs.
    
    Args:
        metric_names: List of metric names to track
    """
    
    def __init__(self, metric_names: list = None):
        self.metric_names = metric_names or ['loss']
        self.history: Dict[str, list] = {name: [] for name in self.metric_names}
        self.best_values: Dict[str, float] = {}
        self.best_epochs: Dict[str, int] = {}
    
    def update(self, metrics: Dict[str, float], epoch: int) -> None:
        """Update tracker with new metric values."""
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
                self.metric_names.append(name)
            self.history[name].append(value)
            
            if name not in self.best_values or self._is_better(name, value):
                self.best_values[name] = value
                self.best_epochs[name] = epoch
    
    def _is_better(self, name: str, value: float) -> bool:
        """Check if new value is better than current best."""
        if np.isnan(value):
            return False
        current_best = self.best_values.get(name, float('nan'))
        if np.isnan(current_best):
            return True
        
        minimize = name in ('loss', 'rmse', 'mae')
        if minimize:
            return value < current_best
        return value > current_best
    
    def get_history(self, name: str) -> list:
        """Get history of a specific metric."""
        return self.history.get(name, [])
    
    def get_best(self, name: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric."""
        return self.best_values.get(name, float('nan')), self.best_epochs.get(name, -1)
    
    def to_dict(self) -> Dict[str, list]:
        """Export history as dictionary."""
        return dict(self.history)
