"""
Dataset classes and samplers for SEAL-ADME training.

Provides utilities for loading graph data and balanced multi-task sampling.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Iterator

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class PretrainDataset:
    """
    Dataset for multi-task pretraining from graph files.
    
    Loads graphs from disk and organizes by task for balanced sampling.
    
    Args:
        graph_dir: Directory containing .pt graph files
        metadata_df: DataFrame with columns: graph_id, task_name, label, split
    """
    
    def __init__(self, graph_dir: Path, metadata_df: pd.DataFrame):
        self.graph_dir = Path(graph_dir)
        self.metadata = metadata_df
        
        # Build task indices
        self.task_to_indices = defaultdict(list)
        self.task_to_labels = defaultdict(list)
        
        for idx, row in metadata_df.iterrows():
            task = row['task_name']
            self.task_to_indices[task].append(idx)
            self.task_to_labels[task].append(row['label'])
        
        self.task_names = list(self.task_to_indices.keys())
        
        logger.info(f"Loaded {len(self.task_names)} tasks:")
        for task in self.task_names:
            n_samples = len(self.task_to_indices[task])
            n_pos = sum(self.task_to_labels[task])
            logger.info(f"  {task}: {n_samples} samples ({n_pos} positive)")
    
    def load_graph(self, idx: int) -> Data:
        """Load a single graph by metadata index."""
        row = self.metadata.iloc[idx]
        graph_path = self.graph_dir / f"{row['graph_id']}.pt"
        graph = torch.load(graph_path, weights_only=False)
        graph.task_name = row['task_name']
        graph.y = torch.FloatTensor([float(row['label'])])
        return graph
    
    def get_task_indices(self, task_name: str, split: str) -> List[int]:
        """Get indices for a task and split."""
        mask = (self.metadata['task_name'] == task_name) & (self.metadata['split'] == split)
        return self.metadata[mask].index.tolist()
    
    def __len__(self) -> int:
        return len(self.metadata)


class GraphListDataset:
    """
    Simple dataset wrapper for list of pre-loaded graphs.
    
    Used for finetuning when graphs are already in memory.
    
    Args:
        graphs: List of PyG Data objects
        task_name: Optional task name to assign
    """
    
    def __init__(self, graphs: List[Data], task_name: Optional[str] = None):
        self.graphs = graphs
        self.task_name = task_name
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


class BalancedMultiTaskSampler:
    """
    Sampler for balanced multi-task training.
    
    Ensures each task is sampled equally regardless of dataset size.
    
    Args:
        dataset: PretrainDataset instance
        split: Data split to sample from ('train', 'valid', 'test')
        batch_size: Batch size
        samples_per_task_per_epoch: Samples per task per epoch (None = min task size)
    """
    
    def __init__(
        self,
        dataset: PretrainDataset,
        split: str,
        batch_size: int,
        samples_per_task_per_epoch: Optional[int] = None
    ):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        
        # Get indices per task for this split
        self.task_indices = {}
        for task in dataset.task_names:
            indices = dataset.get_task_indices(task, split)
            self.task_indices[task] = indices
        
        # Determine samples per task
        if samples_per_task_per_epoch is None:
            samples_per_task_per_epoch = min(
                len(v) for v in self.task_indices.values() if len(v) > 0
            )
        
        self.samples_per_task = samples_per_task_per_epoch
        self.epoch_length = len(dataset.task_names) * samples_per_task_per_epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices."""
        # Create pools with oversampling if needed
        task_pools = {}
        for task, indices in self.task_indices.items():
            if len(indices) == 0:
                continue
            shuffled = np.random.permutation(indices).tolist()
            while len(shuffled) < self.samples_per_task:
                shuffled.extend(np.random.permutation(indices).tolist())
            task_pools[task] = shuffled[:self.samples_per_task]
        
        # Interleave tasks
        all_indices = []
        for i in range(self.samples_per_task):
            for task in self.dataset.task_names:
                if task in task_pools:
                    all_indices.append(task_pools[task][i])
        
        np.random.shuffle(all_indices)
        
        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]
    
    def __len__(self) -> int:
        return (self.epoch_length + self.batch_size - 1) // self.batch_size


class ProportionalTaskSampler:
    """
    Sampler that samples tasks proportionally to dataset size.
    
    Larger datasets get more samples per epoch.
    
    Args:
        task_datasets: Dict mapping task_name -> {'train': [graphs], ...}
        batch_size: Batch size
        split: Which split to sample from
    """
    
    def __init__(
        self,
        task_datasets: Dict[str, Dict[str, List[Data]]],
        batch_size: int,
        split: str = 'train'
    ):
        self.task_datasets = task_datasets
        self.batch_size = batch_size
        self.split = split
        self.task_names = list(task_datasets.keys())
        
        # Compute task probabilities
        total_samples = sum(
            len(task_datasets[t][split]) for t in self.task_names
        )
        self.task_probs = [
            len(task_datasets[t][split]) / total_samples
            for t in self.task_names
        ]
        
        # Total batches per epoch
        self.max_batches = max(
            len(task_datasets[t][split]) // batch_size + 1
            for t in self.task_names
        )
    
    def sample_task(self) -> str:
        """Sample a task according to proportions."""
        return np.random.choice(self.task_names, p=self.task_probs)
    
    def __len__(self) -> int:
        return self.max_batches


def load_task_graphs(
    graph_dir: Path,
    task_name: str,
    splits: List[str] = None
) -> Dict[str, List[Data]]:
    """
    Load graphs for a single task from directory.
    
    Expects files named: {prefix}_{task}_{split}_{idx}.pt
    
    Args:
        graph_dir: Directory containing graph files
        task_name: Name of task
        splits: Which splits to load (default: train, valid, test)
        
    Returns:
        Dict mapping split -> list of graphs
    """
    if splits is None:
        splits = ['train', 'valid', 'test']
    
    graph_dir = Path(graph_dir)
    result = {}
    
    for split in splits:
        graphs = []
        pattern = f"*_{task_name}_{split}_*.pt"
        
        for path in sorted(graph_dir.glob(pattern)):
            try:
                g = torch.load(path, weights_only=False)
                graphs.append(g)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        
        result[split] = graphs
        logger.info(f"  {task_name}/{split}: {len(graphs)} graphs")
    
    return result


def load_all_task_graphs(
    base_dir: Path,
    task_folders: Dict[str, str]
) -> Dict[str, Dict[str, List[Data]]]:
    """
    Load graphs for multiple tasks.
    
    Args:
        base_dir: Base directory containing task folders
        task_folders: Dict mapping task_name -> folder_name
        
    Returns:
        Nested dict: task_name -> split -> list of graphs
    """
    base_dir = Path(base_dir)
    all_datasets = {}
    
    for task_name, folder_name in task_folders.items():
        task_dir = base_dir / folder_name
        if not task_dir.exists():
            logger.warning(f"Task directory not found: {task_dir}")
            continue
        
        logger.info(f"Loading {task_name}...")
        all_datasets[task_name] = load_task_graphs(task_dir, folder_name)
    
    return all_datasets


def create_data_loaders(
    graphs: List[Data],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create PyG DataLoader from graph list.
    
    Args:
        graphs: List of PyG Data objects
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
