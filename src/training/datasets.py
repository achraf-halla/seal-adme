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
    
    Supports directory structures:
    1. Task-organized: graph_dir/{task_name}/features_{task}_{split}_{idx}.pt
    2. Split-organized: graph_dir/{split}/features_{task}_{split}_{idx}.pt  
    3. Flat with metadata: graph_dir/{graph_id}.pt + metadata DataFrame
    
    Args:
        graph_dir: Directory containing graph files
        metadata_df: Optional DataFrame with task/label info
        task_filter: Optional list of task names to include (e.g., pretrain tasks only)
    """
    
    def __init__(
        self, 
        graph_dir: Path, 
        metadata_df: pd.DataFrame = None,
        task_filter: List[str] = None
    ):
        self.graph_dir = Path(graph_dir)
        self.task_filter = task_filter
        
        # Storage for loaded graphs: (task_name, split) -> list of (graph, label)
        self.graphs = {}
        self.task_to_indices = defaultdict(list)
        self.task_to_labels = defaultdict(list)
        
        # Global index mapping: idx -> (task_name, split, local_idx)
        self.index_map = []
        
        if metadata_df is not None and 'graph_id' in metadata_df.columns:
            self._load_from_metadata(metadata_df)
        else:
            self._load_from_directory()
        
        self.task_names = list(self.task_to_indices.keys())
        
        logger.info(f"Loaded {len(self.task_names)} tasks:")
        for task in self.task_names:
            n_samples = len(self.task_to_indices[task])
            n_pos = sum(self.task_to_labels[task])
            logger.info(f"  {task}: {n_samples} samples ({n_pos} positive)")
    
    def _parse_filename(self, filename: str) -> tuple:
        """
        Parse task name and split from filename.
        
        Expected patterns:
        - features_{task}_{split}_{idx}.pt
        - {task}_{split}_{idx}.pt
        - graph_{task}_{split}_{idx}.pt
        
        Returns: (task_name, split) or (None, None) if unparseable
        """
        stem = Path(filename).stem
        parts = stem.split('_')
        
        # Handle various prefixes
        if parts[0] in ('features', 'graph'):
            parts = parts[1:]
        
        if len(parts) < 3:
            return None, None
        
        # Last part is index, second-to-last is split
        split = parts[-2]
        if split not in ('train', 'valid', 'test'):
            # Try finding split elsewhere
            for i, p in enumerate(parts):
                if p in ('train', 'valid', 'test'):
                    split = p
                    task_name = '_'.join(parts[:i])
                    return task_name, split
            return None, None
        
        # Task name is everything before split
        task_name = '_'.join(parts[:-2])
        return task_name, split
    
    def _load_from_metadata(self, metadata_df: pd.DataFrame):
        """Load using metadata DataFrame."""
        self.metadata = metadata_df
        
        for idx, row in metadata_df.iterrows():
            task = row['task_name']
            
            if self.task_filter and task not in self.task_filter:
                continue
            
            split = row.get('split', 'train')
            label = row.get('label', row.get('Y', 0))
            
            self.task_to_indices[task].append(len(self.index_map))
            self.task_to_labels[task].append(label)
            self.index_map.append((task, split, idx, row.get('graph_id', idx)))
    
    def _load_from_directory(self):
        """Load by scanning directory structure."""
        subdirs = [d for d in self.graph_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            for subdir in sorted(subdirs):
                self._load_graphs_from_folder(subdir)
        else:
            self._load_graphs_from_folder(self.graph_dir)
    
    def _load_graphs_from_folder(self, folder: Path):
        """Load all .pt files from a folder, parsing task/split from filenames."""
        graph_files = sorted(folder.glob("*.pt"))
        
        for graph_path in graph_files:
            try:
                # Parse filename
                task_name, split = self._parse_filename(graph_path.name)
                
                # If can't parse from filename, try to get from graph
                if task_name is None:
                    graph = torch.load(graph_path, weights_only=False)
                    task_name = getattr(graph, 'task_name', folder.name)
                    split = 'train'
                else:
                    graph = torch.load(graph_path, weights_only=False)
                
                # Apply task filter
                if self.task_filter and task_name not in self.task_filter:
                    continue
                
                # Get label
                if hasattr(graph, 'y') and graph.y is not None:
                    label = float(graph.y.item()) if graph.y.numel() == 1 else float(graph.y[0].item())
                else:
                    label = 0.0
                
                # Store
                key = (task_name, split)
                if key not in self.graphs:
                    self.graphs[key] = []
                
                local_idx = len(self.graphs[key])
                self.graphs[key].append((graph, label))
                
                # Update indices
                global_idx = len(self.index_map)
                self.task_to_indices[task_name].append(global_idx)
                self.task_to_labels[task_name].append(label)
                self.index_map.append((task_name, split, local_idx, str(graph_path)))
                
            except Exception as e:
                logger.warning(f"Failed to load {graph_path}: {e}")
    
    def load_graph(self, idx: int) -> Data:
        """Load a single graph by global index."""
        task_name, split, local_idx, path_or_id = self.index_map[idx]
        
        key = (task_name, split)
        if key in self.graphs:
            graph, label = self.graphs[key][local_idx]
            graph = graph.clone()
        else:
            if isinstance(path_or_id, str) and Path(path_or_id).exists():
                graph_path = Path(path_or_id)
            else:
                graph_path = self.graph_dir / f"{path_or_id}.pt"
            
            graph = torch.load(graph_path, weights_only=False)
            label = self.task_to_labels[task_name][self.task_to_indices[task_name].index(idx)]
        
        graph.task_name = task_name
        graph.y = torch.FloatTensor([float(label)])
        return graph
    
    def get_task_indices(self, task_name: str, split: str = None) -> List[int]:
        """Get global indices for a task (optionally filtered by split)."""
        indices = []
        for global_idx in self.task_to_indices.get(task_name, []):
            if split is None:
                indices.append(global_idx)
            else:
                t, s, _, _ = self.index_map[global_idx]
                if s == split:
                    indices.append(global_idx)
        return indices
    
    def __len__(self) -> int:
        return len(self.index_map)


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
        split: Data split to sample from ('train', 'valid', 'test') or None for all
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
            if len(indices) > 0:
                self.task_indices[task] = indices
        
        self.active_tasks = list(self.task_indices.keys())
        
        if not self.active_tasks:
            raise ValueError(f"No tasks found for split '{split}'")
        
        # Determine samples per task
        if samples_per_task_per_epoch is None:
            samples_per_task_per_epoch = min(
                len(v) for v in self.task_indices.values()
            )
        
        self.samples_per_task = samples_per_task_per_epoch
        self.epoch_length = len(self.active_tasks) * samples_per_task_per_epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices."""
        # Create pools with oversampling if needed
        task_pools = {}
        for task, indices in self.task_indices.items():
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


def pad_s_matrix(data: Data, max_fragments: int = None) -> Data:
    """
    Pad the fragment membership matrix (s) to consistent size.
    
    The 's' matrix has shape [n_atoms, n_fragments] but n_fragments varies.
    This pads the second dimension to max_fragments.
    
    Args:
        data: PyG Data object with 's' attribute
        max_fragments: Target number of fragment columns (None = no padding)
        
    Returns:
        Data object with padded 's' matrix
    """
    if not hasattr(data, 's') or data.s is None:
        return data
    
    if max_fragments is None:
        return data
    
    s = data.s
    n_atoms, n_frags = s.shape
    
    if n_frags < max_fragments:
        # Pad with zeros
        padding = torch.zeros(n_atoms, max_fragments - n_frags, dtype=s.dtype)
        data.s = torch.cat([s, padding], dim=1)
    elif n_frags > max_fragments:
        # Truncate (shouldn't happen normally)
        data.s = s[:, :max_fragments]
    
    return data


def collate_with_padding(data_list: List[Data]) -> Data:
    """
    Custom collate function that pads 's' matrices before batching.
    
    Args:
        data_list: List of Data objects
        
    Returns:
        Batched Data object
    """
    from torch_geometric.data import Batch
    
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


class PaddedDataLoader(DataLoader):
    """
    DataLoader that pads fragment matrices for consistent batching.
    
    Wraps PyG DataLoader with custom collate function.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs
    ):
        # Remove collate_fn if provided, we'll use our own
        kwargs.pop('collate_fn', None)
        
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_with_padding,
            **kwargs
        )
