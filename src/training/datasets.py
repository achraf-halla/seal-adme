"""
Dataset and sampling utilities for multitask training.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class MultiTaskDataset:
    """
    Dataset for multitask learning with per-task indexing.
    
    Manages graph data across multiple tasks with metadata for
    balanced sampling during training.
    
    Args:
        graph_dir: Directory containing .pt graph files
        metadata_df: DataFrame with columns:
            - graph_id: Filename stem of graph file
            - task_name: Task this sample belongs to
            - label: Target value
            - split: 'train', 'valid', or 'test'
    """
    
    def __init__(self, graph_dir: Path, metadata_df: pd.DataFrame):
        self.graph_dir = Path(graph_dir)
        self.metadata = metadata_df
        
        # Build task-to-indices mapping
        self.task_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.task_to_labels: Dict[str, List[float]] = defaultdict(list)
        
        for idx, row in metadata_df.iterrows():
            task = row['task_name']
            self.task_to_indices[task].append(idx)
            self.task_to_labels[task].append(row['label'])
        
        self.task_names = list(self.task_to_indices.keys())
        
        # Log dataset statistics
        logger.info(f"Loaded dataset with {len(self.task_names)} tasks:")
        for task in self.task_names:
            n_samples = len(self.task_to_indices[task])
            labels = self.task_to_labels[task]
            n_pos = sum(1 for l in labels if l > 0.5)
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
        """Get indices for a specific task and split."""
        mask = (
            (self.metadata['task_name'] == task_name) &
            (self.metadata['split'] == split)
        )
        return self.metadata[mask].index.tolist()
    
    def get_task_graphs(
        self,
        task_name: str,
        split: str = None
    ) -> List[Data]:
        """Load all graphs for a task (optionally filtered by split)."""
        if split:
            indices = self.get_task_indices(task_name, split)
        else:
            indices = self.task_to_indices[task_name]
        
        return [self.load_graph(i) for i in indices]
    
    def __len__(self) -> int:
        return len(self.metadata)


class TaskGraphDataset:
    """
    Simple container for pre-loaded graphs organized by split.
    
    Args:
        train: List of training graphs
        valid: List of validation graphs
        test: List of test graphs
    """
    
    def __init__(
        self,
        train: List[Data] = None,
        valid: List[Data] = None,
        test: List[Data] = None
    ):
        self.train = train or []
        self.valid = valid or []
        self.test = test or []
    
    def __getitem__(self, split: str) -> List[Data]:
        return getattr(self, split, [])
    
    def get(self, split: str, default=None) -> List[Data]:
        return getattr(self, split, default)


class BalancedMultiTaskSampler:
    """
    Sampler that balances samples across tasks during training.
    
    Ensures each task contributes equally to each epoch, regardless
    of dataset size. Smaller datasets are oversampled.
    
    Args:
        dataset: MultiTaskDataset instance
        split: Which split to sample from ('train', 'valid', 'test')
        batch_size: Batch size
        samples_per_task: Samples per task per epoch (None = use minimum)
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
        
        # Get indices for each task in this split
        self.task_indices: Dict[str, List[int]] = {}
        for task in dataset.task_names:
            indices = dataset.get_task_indices(task, split)
            if indices:
                self.task_indices[task] = indices
        
        # Determine samples per task
        if samples_per_task is None:
            samples_per_task = min(len(v) for v in self.task_indices.values())
        
        self.samples_per_task = samples_per_task
        self.epoch_length = len(self.task_indices) * samples_per_task
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batch indices."""
        # Create shuffled pool for each task (with oversampling if needed)
        task_pools = {}
        for task, indices in self.task_indices.items():
            shuffled = np.random.permutation(indices).tolist()
            
            # Oversample if needed
            while len(shuffled) < self.samples_per_task:
                shuffled.extend(np.random.permutation(indices).tolist())
            
            task_pools[task] = shuffled[:self.samples_per_task]
        
        # Interleave tasks
        all_indices = []
        for i in range(self.samples_per_task):
            for task in self.task_indices.keys():
                all_indices.append(task_pools[task][i])
        
        # Shuffle all indices
        np.random.shuffle(all_indices)
        
        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]
    
    def __len__(self) -> int:
        return (self.epoch_length + self.batch_size - 1) // self.batch_size


def load_task_datasets(
    base_dir: Path,
    task_configs: List[tuple] = None
) -> Dict[str, TaskGraphDataset]:
    """
    Load graph datasets for multiple tasks.
    
    Args:
        base_dir: Base directory containing task subdirectories
        task_configs: List of (task_name, folder_name) tuples
        
    Returns:
        Dictionary mapping task names to TaskGraphDataset instances
    """
    if task_configs is None:
        task_configs = [
            ("solubility_aqsoldb", "solubility_aqsoldb"),
            ("caco2", "caco2"),
            ("half_life_obach", "half_life_obach"),
            ("AKB", "AKB"),
            ("AKA", "AKA")
        ]
    
    base_dir = Path(base_dir)
    task_datasets = {}
    
    for task_name, folder_name in task_configs:
        data_dir = base_dir / folder_name
        
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            continue
        
        def load_split_graphs(split_name: str) -> List[Data]:
            graphs = []
            pattern = f"*_{split_name}_*.pt"
            
            for file in sorted(data_dir.glob(pattern)):
                try:
                    g = torch.load(file, weights_only=False)
                    graphs.append(g)
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")
            
            return graphs
        
        dataset = TaskGraphDataset(
            train=load_split_graphs("train"),
            valid=load_split_graphs("valid"),
            test=load_split_graphs("test")
        )
        
        task_datasets[task_name] = dataset
        logger.info(
            f"Loaded {task_name}: "
            f"train={len(dataset.train)}, "
            f"valid={len(dataset.valid)}, "
            f"test={len(dataset.test)}"
        )
    
    return task_datasets


def collate_by_task(
    graphs: List[Data]
) -> Dict[str, List[Data]]:
    """
    Group graphs by their task_name attribute.
    
    Args:
        graphs: List of Data objects with task_name attribute
        
    Returns:
        Dictionary mapping task names to graph lists
    """
    task_batches = defaultdict(list)
    for g in graphs:
        task_batches[g.task_name].append(g)
    return dict(task_batches)
