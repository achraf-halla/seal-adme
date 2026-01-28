"""
Dataset classes and samplers for multi-task training.

This module provides dataset wrappers and balanced samplers for
training SEAL models on multiple ADME prediction tasks.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class MultiTaskDataset:
    """
    Dataset for multi-task pretraining with metadata-based indexing.
    
    Loads graph files on-demand based on a metadata DataFrame that
    specifies graph IDs, task names, labels, and splits.
    
    Args:
        graph_dir: Directory containing .pt graph files
        metadata_df: DataFrame with columns [graph_id, task_name, label, split]
    """
    
    def __init__(self, graph_dir: Union[str, Path], metadata_df: pd.DataFrame):
        self.graph_dir = Path(graph_dir)
        self.metadata = metadata_df.reset_index(drop=True)
        
        self.task_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.task_to_labels: Dict[str, List[float]] = defaultdict(list)
        
        for idx, row in self.metadata.iterrows():
            task = row['task_name']
            self.task_to_indices[task].append(idx)
            self.task_to_labels[task].append(float(row['label']))
        
        self.task_names = list(self.task_to_indices.keys())
        
        logger.info(f"Loaded dataset with {len(self.task_names)} tasks:")
        for task in self.task_names:
            n_samples = len(self.task_to_indices[task])
            labels = self.task_to_labels[task]
            n_pos = sum(1 for l in labels if l > 0.5)
            logger.info(f"  {task}: {n_samples} samples ({n_pos} positive)")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def load_graph(self, idx: int) -> Data:
        """Load a single graph by metadata index."""
        row = self.metadata.iloc[idx]
        graph_path = self.graph_dir / f"{row['graph_id']}.pt"
        graph = torch.load(graph_path, weights_only=False)
        graph.task_name = row['task_name']
        graph.y = torch.tensor([float(row['label'])], dtype=torch.float32)
        return graph
    
    def get_task_indices(self, task_name: str, split: str) -> List[int]:
        """Get indices for a specific task and split."""
        mask = (
            (self.metadata['task_name'] == task_name) & 
            (self.metadata['split'] == split)
        )
        return self.metadata[mask].index.tolist()
    
    def get_task_graphs(
        self,
        task_name: str,
        split: str,
        max_samples: Optional[int] = None
    ) -> List[Data]:
        """Load all graphs for a task and split."""
        indices = self.get_task_indices(task_name, split)
        if max_samples is not None:
            indices = indices[:max_samples]
        return [self.load_graph(i) for i in indices]


class TaskGraphDataset:
    """
    Simple container for pre-loaded task graphs organized by split.
    
    Used for finetuning where graphs are loaded once and reused.
    
    Args:
        task_name: Name of the task
        train_graphs: List of training graphs
        valid_graphs: List of validation graphs
        test_graphs: List of test graphs
    """
    
    def __init__(
        self,
        task_name: str,
        train_graphs: List[Data],
        valid_graphs: List[Data],
        test_graphs: List[Data]
    ):
        self.task_name = task_name
        self.splits = {
            'train': train_graphs,
            'valid': valid_graphs,
            'test': test_graphs
        }
    
    def __getitem__(self, split: str) -> List[Data]:
        return self.splits[split]
    
    def __len__(self) -> int:
        return sum(len(graphs) for graphs in self.splits.values())
    
    @property
    def train(self) -> List[Data]:
        return self.splits['train']
    
    @property
    def valid(self) -> List[Data]:
        return self.splits['valid']
    
    @property
    def test(self) -> List[Data]:
        return self.splits['test']


class BalancedMultiTaskSampler:
    """
    Balanced sampler for multi-task training.
    
    Ensures each task is sampled approximately equally during training
    by cycling through tasks and sampling from each.
    
    Args:
        dataset: MultiTaskDataset instance
        split: Split to sample from ('train', 'valid', 'test')
        batch_size: Number of samples per batch
        samples_per_task: Samples per task per epoch (None = min task size)
    """
    
    def __init__(
        self,
        dataset: MultiTaskDataset,
        split: str,
        batch_size: int,
        samples_per_task: Optional[int] = None
    ):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        
        self.task_indices: Dict[str, List[int]] = {}
        for task in dataset.task_names:
            indices = dataset.get_task_indices(task, split)
            self.task_indices[task] = indices
        
        if samples_per_task is None:
            samples_per_task = min(
                len(indices) for indices in self.task_indices.values()
            )
        
        self.samples_per_task = samples_per_task
        self.epoch_length = len(dataset.task_names) * samples_per_task
    
    def __iter__(self) -> Iterator[List[int]]:
        task_pools = {}
        for task, indices in self.task_indices.items():
            shuffled = np.random.permutation(indices).tolist()
            while len(shuffled) < self.samples_per_task:
                shuffled.extend(np.random.permutation(indices).tolist())
            task_pools[task] = shuffled[:self.samples_per_task]
        
        all_indices = []
        for i in range(self.samples_per_task):
            for task in self.dataset.task_names:
                all_indices.append(task_pools[task][i])
        
        np.random.shuffle(all_indices)
        
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]
    
    def __len__(self) -> int:
        return (self.epoch_length + self.batch_size - 1) // self.batch_size


def load_task_graphs(
    base_dir: Union[str, Path],
    task_configs: List[tuple],
    file_pattern: str = "features_{task}_{split}_*.pt"
) -> Dict[str, TaskGraphDataset]:
    """
    Load pre-computed graphs for multiple tasks.
    
    Args:
        base_dir: Base directory containing task subdirectories
        task_configs: List of (task_name, folder_name) tuples
        file_pattern: Pattern for graph files (uses task and split)
        
    Returns:
        Dictionary mapping task names to TaskGraphDataset instances
    """
    base_path = Path(base_dir)
    task_datasets = {}
    
    for task_name, folder_name in task_configs:
        data_dir = base_path / folder_name
        
        splits = {}
        for split in ['train', 'valid', 'test']:
            pattern = file_pattern.format(task=folder_name, split=split)
            graphs = []
            for file in sorted(data_dir.glob(pattern)):
                g = torch.load(file, weights_only=False)
                graphs.append(g)
            splits[split] = graphs
        
        task_datasets[task_name] = TaskGraphDataset(
            task_name=task_name,
            train_graphs=splits['train'],
            valid_graphs=splits['valid'],
            test_graphs=splits['test']
        )
        
        logger.info(
            f"Loaded {task_name}: "
            f"train={len(splits['train'])}, "
            f"valid={len(splits['valid'])}, "
            f"test={len(splits['test'])}"
        )
    
    return task_datasets


def create_data_loader(
    graphs: List[Data],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """Create a PyG DataLoader from a list of graphs."""
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )


def collate_by_task(graphs: List[Data]) -> Dict[str, List[Data]]:
    """
    Group graphs by their task_name attribute.
    
    Useful for multi-task training where batches contain
    samples from multiple tasks.
    
    Args:
        graphs: List of Data objects with task_name attribute
        
    Returns:
        Dictionary mapping task names to graph lists
    """
    task_batches: Dict[str, List[Data]] = defaultdict(list)
    for g in graphs:
        task_name = getattr(g, 'task_name', 'default')
        task_batches[task_name].append(g)
    return dict(task_batches)
