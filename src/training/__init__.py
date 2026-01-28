"""
SEAL-ADME Training Module.

This module provides training infrastructure for SEAL models,
including datasets, samplers, metrics, and training loops.

Datasets:
    MultiTaskDataset: Dataset with metadata-based indexing
    TaskGraphDataset: Pre-loaded graphs organized by split
    BalancedMultiTaskSampler: Balanced sampling across tasks

Metrics:
    compute_classification_metrics: AUROC, AUPRC
    compute_regression_metrics: Spearman, Pearson, RMSE, MAE, R²
    MetricTracker: Track metrics over training

Trainers:
    PretrainTrainer: Multi-task classification pretraining
    FinetuneTrainer: Multi-task regression finetuning
"""

from .datasets import (
    MultiTaskDataset,
    TaskGraphDataset,
    BalancedMultiTaskSampler,
    load_task_graphs,
    create_data_loader,
    collate_by_task,
)

from .metrics import (
    safe_auroc,
    safe_auprc,
    safe_spearman,
    safe_pearson,
    safe_rmse,
    safe_mae,
    safe_r2,
    compute_classification_metrics,
    compute_regression_metrics,
    aggregate_task_metrics,
    MetricTracker,
)

from .trainers import (
    PretrainTrainer,
    FinetuneTrainer,
)


__all__ = [
    # Datasets
    "MultiTaskDataset",
    "TaskGraphDataset",
    "BalancedMultiTaskSampler",
    "load_task_graphs",
    "create_data_loader",
    "collate_by_task",
    # Metrics
    "safe_auroc",
    "safe_auprc",
    "safe_spearman",
    "safe_pearson",
    "safe_rmse",
    "safe_mae",
    "safe_r2",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "aggregate_task_metrics",
    "MetricTracker",
    # Trainers
    "PretrainTrainer",
    "FinetuneTrainer",
]
