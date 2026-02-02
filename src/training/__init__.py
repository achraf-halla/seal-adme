"""
SEAL-ADME Training Module.
Provides:
- Dataset classes for pretraining and finetuning
- Balanced multi-task sampling
- Pretraining trainer for classification (AUROC/AUPRC)
- Finetuning trainer for regression (Spearman/Pearson/RMSE)
"""

from .datasets import (
    PretrainDataset,
    FinetuneDataset,
    BalancedMultiTaskSampler,
    collate_with_padding,
    create_dataloader,
    load_pretrain_dataset,
    load_finetune_datasets,
)

from .pretrain import PretrainTrainer

from .finetune import (
    MultiTaskFinetuneTrainer,
    SingleTaskFinetuneTrainer,
    spearman_scorer,
    pearson_scorer,
)

__all__ = [
    # Datasets
    "PretrainDataset",
    "FinetuneDataset",
    "BalancedMultiTaskSampler",
    "collate_with_padding",
    "create_dataloader",
    "load_pretrain_dataset",
    "load_finetune_datasets",
    # Trainers
    "PretrainTrainer",
    "MultiTaskFinetuneTrainer",
    "SingleTaskFinetuneTrainer",
    # Utilities
    "spearman_scorer",
    "pearson_scorer",
]
