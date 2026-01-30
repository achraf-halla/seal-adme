"""
Dataset classes and samplers for SEAL-ADME training.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Iterator

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def collate_with_padding(data_list: List[Data]) -> Batch:
    """
    Custom collate function that pads 's' matrices before batching.
    
    The fragment membership matrix 's' has variable columns per molecule.
    This pads all matrices to the same size for batching.
    """
    if not data_list:
        return Batch.from_data_list([])
    
    # Find max fragments across all graphs
    max_frags = 1
    for data in data_list:
        if hasattr(data, 's') and data.s is not None:
            max_frags = max(max_frags, data.s.shape[1])
    
    # Pad all s matrices
    padded_list = []
    for data in data_list:
        data_copy = data.clone()
        if hasattr(data_copy, 's') and data_copy.s is not None:
            n_atoms, n_frags = data_copy.s.shape
            if n_frags < max_frags:
                padding = torch.zeros(n_atoms, max_frags - n_frags, dtype=data_copy.s.dtype)
                data_copy.s = torch.cat([data_copy.s, padding], dim=1)
        padded_list.append(data_copy)
    
    return Batch.from_data_list(padded_list)


class PretrainDataset:
    """
    Dataset for multi-task pretraining from graph files.
    
    Loads all graphs from pretrain directory and organizes by task_name.
    """
    
    def __init__(self, graph_dir: Path):
        """
        Args:
            graph_dir: Directory containing pretrain graphs (e.g., data/graphs/pretrain)
        """
        self.graph_dir = Path(graph_dir)
        
        # Load all graphs
        self.graphs = []
        self.task_to_indices = defaultdict(list)
        self.task_to_labels = defaultdict(list)
        
        self._load_graphs()
        
        self.task_names = list(self.task_to_indices.keys())
        
        logger.info(f"Loaded {len(self.task_names)} pretrain tasks:")
        for task in sorted(self.task_names):
            n = len(self.task_to_indices[task])
            pos = sum(self.task_to_labels[task])
            logger.info(f"  {task}: {n} samples ({pos:.0f} positive)")
    
    def _load_graphs(self):
        """Load all graphs from directory."""
        graph_files = sorted(self.graph_dir.glob("*.pt"))
        
        for i, path in enumerate(graph_files):
            try:
                g = torch.load(path, weights_only=False)
                
                # Get task_name from graph attribute
                task_name = getattr(g, 'task_name', 'unknown')
                
                # Get label
                label = float(g.y.item()) if hasattr(g, 'y') and g.y is not None else 0.0
                
                self.graphs.append(g)
                self.task_to_indices[task_name].append(i)
                self.task_to_labels[task_name].append(label)
                
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    
    def get_task_indices(self, task_name: str) -> List[int]:
        """Get indices for a specific task."""
        return self.task_to_indices.get(task_name, [])
    
    def get_task_graphs(self, task_name: str) -> List[Data]:
        """Get all graphs for a specific task."""
        indices = self.get_task_indices(task_name)
        return [self.graphs[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


class FinetuneDataset:
    """
    Dataset for finetuning from task-organized graph directories.
    
    Expects structure: finetune/{task_name}/{split}/*.pt
    """
    
    def __init__(self, graph_dir: Path, task_name: str):
        """
        Args:
            graph_dir: Base finetune directory (e.g., data/graphs/finetune)
            task_name: Name of the task to load
        """
        self.graph_dir = Path(graph_dir)
        self.task_name = task_name
        self.task_dir = self.graph_dir / task_name
        
        # Load graphs for each split
        self.splits = {}
        for split in ['train', 'valid', 'test']:
            split_dir = self.task_dir / split
            if split_dir.exists():
                graphs = self._load_split(split_dir)
                self.splits[split] = graphs
                logger.info(f"  {task_name}/{split}: {len(graphs)} graphs")
    
    def _load_split(self, split_dir: Path) -> List[Data]:
        """Load all graphs from a split directory."""
        graphs = []
        for path in sorted(split_dir.glob("*.pt")):
            try:
                g = torch.load(path, weights_only=False)
                graphs.append(g)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        return graphs
    
    def get_split(self, split: str) -> List[Data]:
        """Get graphs for a specific split."""
        return self.splits.get(split, [])
    
    @property
    def train(self) -> List[Data]:
        return self.get_split('train')
    
    @property
    def valid(self) -> List[Data]:
        return self.get_split('valid')
    
    @property
    def test(self) -> List[Data]:
        return self.get_split('test')


class BalancedMultiTaskSampler:
    """
    Sampler for balanced multi-task training.
    
    Ensures each task is sampled equally regardless of dataset size.
    """
    
    def __init__(
        self,
        dataset: PretrainDataset,
        batch_size: int,
        samples_per_task_per_epoch: Optional[int] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get task sizes
        self.task_indices = {
            task: dataset.get_task_indices(task)
            for task in dataset.task_names
        }
        
        # Filter out empty tasks
        self.active_tasks = [t for t in dataset.task_names if len(self.task_indices[t]) > 0]
        
        if not self.active_tasks:
            raise ValueError("No tasks with samples found!")
        
        # Samples per task per epoch
        if samples_per_task_per_epoch is None:
            samples_per_task_per_epoch = min(
                len(self.task_indices[t]) for t in self.active_tasks
            )
        
        self.samples_per_task = samples_per_task_per_epoch
        self.epoch_length = len(self.active_tasks) * samples_per_task_per_epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        # Create pools with oversampling if needed
        task_pools = {}
        for task in self.active_tasks:
            indices = self.task_indices[task]
            shuffled = np.random.permutation(indices).tolist()
            while len(shuffled) < self.samples_per_task:
                shuffled.extend(np.random.permutation(indices).tolist())
            task_pools[task] = shuffled[:self.samples_per_task]
        
        # Interleave tasks
        all_indices = []
        for i in range(self.samples_per_task):
            for task in self.active_tasks:
                all_indices.append(task_pools[task][i])
        
        np.random.shuffle(all_indices)
        
        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]
    
    def __len__(self) -> int:
        return (self.epoch_length + self.batch_size - 1) // self.batch_size


def create_dataloader(
    graphs: List[Data],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader with padding collate function."""
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_with_padding,
        pin_memory=torch.cuda.is_available()
    )


def load_pretrain_dataset(graph_dir: Path) -> PretrainDataset:
    """Load pretrain dataset from graph directory."""
    pretrain_dir = Path(graph_dir) / "pretrain"
    if not pretrain_dir.exists():
        raise FileNotFoundError(f"Pretrain directory not found: {pretrain_dir}")
    return PretrainDataset(pretrain_dir)


def load_finetune_datasets(graph_dir: Path) -> Dict[str, FinetuneDataset]:
    """Load all finetune datasets."""
    finetune_dir = Path(graph_dir) / "finetune"
    if not finetune_dir.exists():
        raise FileNotFoundError(f"Finetune directory not found: {finetune_dir}")
    
    datasets = {}
    for task_dir in finetune_dir.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name
            logger.info(f"Loading finetune task: {task_name}")
            datasets[task_name] = FinetuneDataset(finetune_dir, task_name)
    
    return datasets
