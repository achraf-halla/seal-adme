"""
SEAL-ADME Training Module.

Provides trainers for multi-task pretraining (classification) and
finetuning (regression), along with dataset utilities.
"""

from .datasets import (
    PretrainDataset,
    GraphListDataset,
    BalancedMultiTaskSampler,
    ProportionalTaskSampler,
    load_task_graphs,
    load_all_task_graphs,
    create_data_loaders,
)

from .pretrain import PretrainTrainer

from .finetune import (
    FinetuneTrainer,
    spearman_correlation,
    pearson_correlation,
)

__all__ = [
    # Datasets
    "PretrainDataset",
    "GraphListDataset",
    "BalancedMultiTaskSampler",
    "ProportionalTaskSampler",
    "load_task_graphs",
    "load_all_task_graphs",
    "create_data_loaders",
    # Trainers
    "PretrainTrainer",
    "FinetuneTrainer",
    # Utilities
    "spearman_correlation",
    "pearson_correlation",
]
