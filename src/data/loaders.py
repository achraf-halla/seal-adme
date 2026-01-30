"""
Data loaders for TDC ADME datasets.

Uses TDC's native scaffold split for finetuning tasks.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .constants import PRETRAIN_TASKS, FINETUNE_TASKS

logger = logging.getLogger(__name__)


class TDCLoader:
    """Load ADME datasets from Therapeutics Data Commons (TDC)."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize TDC loader.
        
        Args:
            output_dir: Optional directory to cache data
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pretrain_tasks(self) -> Dict[str, pd.DataFrame]:
        """
        Load all pretraining (classification) tasks.
        
        No splitting - returns full dataset for each task.
        
        Returns:
            Dict mapping task_name -> DataFrame with columns [Drug_ID, Drug, Y]
        """
        try:
            from tdc.single_pred import ADME
        except ImportError:
            raise ImportError("pytdc is required. Install with: pip install pytdc")
        
        task_data = {}
        
        for task_name in PRETRAIN_TASKS:
            logger.info(f"Loading pretrain task: {task_name}")
            try:
                data = ADME(name=task_name)
                df = data.get_data()
                df['task_name'] = task_name
                task_data[task_name] = df
                logger.info(f"  Loaded {len(df)} samples")
            except Exception as e:
                logger.warning(f"Failed to load {task_name}: {e}")
        
        return task_data
    
    def load_finetune_tasks(
        self,
        seed: int = 42,
        frac: List[float] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load finetuning (regression) tasks with TDC scaffold split.
        
        Args:
            seed: Random seed for splitting
            frac: Train/valid/test fractions (default: [0.7, 0.1, 0.2])
            
        Returns:
            Nested dict: {task_name: {'train': df, 'valid': df, 'test': df}}
        """
        try:
            from tdc.single_pred import ADME
        except ImportError:
            raise ImportError("pytdc is required. Install with: pip install pytdc")
        
        if frac is None:
            frac = [0.7, 0.1, 0.2]
        
        task_splits = {}
        
        for task_name in FINETUNE_TASKS:
            logger.info(f"Loading finetune task: {task_name}")
            try:
                data = ADME(name=task_name)
                
                # Use TDC's native scaffold split
                split = data.get_split(method='scaffold', seed=seed, frac=frac)
                
                # Add task_name to each split
                for split_name in ['train', 'valid', 'test']:
                    split[split_name]['task_name'] = task_name
                
                task_splits[task_name] = split
                
                logger.info(
                    f"  train={len(split['train'])}, "
                    f"valid={len(split['valid'])}, "
                    f"test={len(split['test'])}"
                )
            except Exception as e:
                logger.warning(f"Failed to load {task_name}: {e}")
        
        return task_splits
    
    def compute_normalization_stats(
        self,
        task_splits: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and std from training data for each task.
        
        Args:
            task_splits: Output from load_finetune_tasks()
            
        Returns:
            Dict mapping task_name -> {'mean': float, 'std': float}
        """
        stats = {}
        
        for task_name, splits in task_splits.items():
            train_df = splits['train']
            y_mean = train_df['Y'].mean()
            y_std = train_df['Y'].std()
            
            # Avoid division by zero
            if y_std == 0 or pd.isna(y_std):
                y_std = 1.0
            
            stats[task_name] = {
                'mean': float(y_mean),
                'std': float(y_std)
            }
            
            logger.info(f"  {task_name}: mean={y_mean:.4f}, std={y_std:.4f}")
        
        return stats


def load_pretrain_data() -> Dict[str, pd.DataFrame]:
    """Convenience function to load all pretrain tasks."""
    loader = TDCLoader()
    return loader.load_pretrain_tasks()


def load_finetune_data(
    seed: int = 42,
    frac: List[float] = None
) -> tuple:
    """
    Convenience function to load finetune tasks with splits and stats.
    
    Returns:
        Tuple of (task_splits, normalization_stats)
    """
    loader = TDCLoader()
    task_splits = loader.load_finetune_tasks(seed=seed, frac=frac)
    stats = loader.compute_normalization_stats(task_splits)
    return task_splits, stats
