"""
Finetuning trainer for regression tasks.

Trains task-specific heads on top of a pretrained encoder.

Features:
- MSE loss with regularization
- Spearman/Pearson/RMSE evaluation
- LR scheduler with patience
- Early stopping
- AMP (mixed precision) training
- Round-robin or proportional task sampling
- Saves predictions to numpy files
"""

import copy
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr

from .datasets import FinetuneDataset, collate_with_padding

logger = logging.getLogger(__name__)


def spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation, handling NaN."""
    if len(y_true) < 2:
        return float('nan')
    r = spearmanr(y_true, y_pred).correlation
    return float('nan') if np.isnan(r) else float(r)


def pearson_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation, handling NaN."""
    if len(y_true) < 2:
        return float('nan')
    r, _ = pearsonr(y_true, y_pred)
    return float('nan') if np.isnan(r) else float(r)


class MultiTaskFinetuneTrainer:
    """
    Trainer for multi-task regression finetuning.
    
    Supports training on multiple regression tasks simultaneously
    with round-robin or proportional task sampling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_datasets: Dict[str, FinetuneDataset],
        norm_stats: Dict[str, Dict[str, float]] = None,
        device: str = None,
        out_dir: Path = None
    ):
        """
        Args:
            model: MultiTaskRegressionModel
            task_datasets: Dict of task_name -> FinetuneDataset
            norm_stats: Dict of task_name -> {'mean': float, 'std': float}
            device: Device to train on
            out_dir: Directory for saving checkpoints
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        self.norm_stats = norm_stats or {t: {'mean': 0.0, 'std': 1.0} for t in self.task_names}
        
        self.out_dir = Path(out_dir) if out_dir else Path("checkpoints/finetune")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info(f"MultiTaskFinetuneTrainer initialized with {len(self.task_names)} tasks:")
        for task_name in self.task_names:
            ds = task_datasets[task_name]
            logger.info(
                f"  {task_name:25s} - Train: {len(ds.train):5d}, "
                f"Valid: {len(ds.valid):4d}, Test: {len(ds.test):4d}"
            )
        logger.info("=" * 60)
    
    def _predict_list(
        self,
        task_name: str,
        graphs: List,
        batch_size: int = 128
    ) -> tuple:
        """Predict on a list of graphs."""
        self.model.eval()
        
        if len(graphs) == 0:
            return np.array([]), np.array([])
        
        loader = DataLoader(
            graphs, batch_size=batch_size, shuffle=False,
            collate_fn=collate_with_padding
        )
        
        preds_list, trues_list = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch, task_name)
                preds = out['output'].view(-1).cpu().numpy()
                trues = batch.y.view(-1).cpu().numpy()
                preds_list.append(preds)
                trues_list.append(trues)
        
        return np.concatenate(trues_list), np.concatenate(preds_list)
    
    def _evaluate_all_tasks(self, batch_size: int = 128) -> Dict:
        """Evaluate Spearman/RMSE on all tasks and splits."""
        metrics = {}
        
        for task_name in self.task_names:
            task_metrics = {}
            ds = self.task_datasets[task_name]
            
            for split, graphs in [('train', ds.train), ('valid', ds.valid), ('test', ds.test)]:
                if len(graphs) == 0:
                    task_metrics[split] = {
                        'spearman': float('nan'),
                        'rmse': float('nan'),
                        'pearson': float('nan')
                    }
                    continue
                
                y_true, y_pred = self._predict_list(task_name, graphs, batch_size)
                
                # Metrics in normalized space
                spearman = spearman_scorer(y_true, y_pred)
                pearson = pearson_scorer(y_true, y_pred)
                rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else float('nan')
                
                # Denormalized RMSE
                std = self.norm_stats[task_name]['std']
                rmse_denorm = rmse * std
                
                task_metrics[split] = {
                    'spearman': spearman,
                    'pearson': pearson,
                    'rmse': rmse,
                    'rmse_denorm': rmse_denorm
                }
            
            metrics[task_name] = task_metrics
        
        return metrics
    
    def train(
        self,
        epochs: int = 120,
        batch_size: int = 64,
        lr: float = 3e-4,
        weight_decay: float = 1e-6,
        mse_weight: float = 1.0,
        grad_clip: float = 1.0,
        lr_patience: int = 8,
        early_stop_patience: int = 25,
        task_sampling: str = 'round_robin',
        num_workers: int = 0
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            epochs: Maximum epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            mse_weight: Weight for MSE loss
            grad_clip: Gradient clipping norm
            lr_patience: Patience for LR scheduler
            early_stop_patience: Patience for early stopping
            task_sampling: 'round_robin' or 'proportional'
            num_workers: DataLoader workers
            
        Returns:
            Training results dict
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=lr_patience, factor=0.5
        )
        
        # AMP setup
        use_amp = torch.cuda.is_available()
        scaler = torch.amp.GradScaler() if use_amp else None
        
        # Create data loaders for each task
        task_loaders = {}
        task_iters = {}
        for task_name in self.task_names:
            train_graphs = self.task_datasets[task_name].train
            loader = DataLoader(
                train_graphs, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, collate_fn=collate_with_padding,
                pin_memory=torch.cuda.is_available()
            )
            task_loaders[task_name] = loader
            task_iters[task_name] = iter(loader)
        
        # Task sampling probabilities
        if task_sampling == 'proportional':
            total = sum(len(self.task_datasets[t].train) for t in self.task_names)
            task_probs = [len(self.task_datasets[t].train) / total for t in self.task_names]
        else:
            task_probs = [1.0 / len(self.task_names)] * len(self.task_names)
        
        logger.info(f"Starting finetuning with {task_sampling} task sampling")
        logger.info(f"Task probabilities: {dict(zip(self.task_names, task_probs))}")
        
        best_state = None
        best_avg_val_spearman = -999.0
        best_epoch = -1
        epochs_no_improve = 0
        
        history = {
            'train_loss': [],
            'task_train_loss': {task: [] for task in self.task_names},
            'avg_val_spearman': []
        }
        
        loss_fn = nn.MSELoss()
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            task_losses = defaultdict(float)
            task_batch_counts = defaultdict(int)
            
            max_batches = max(len(task_loaders[task]) for task in self.task_names)
            
            for batch_idx in range(max_batches):
                # Select task
                if task_sampling == 'round_robin':
                    task_name = self.task_names[batch_idx % len(self.task_names)]
                else:
                    task_name = np.random.choice(self.task_names, p=task_probs)
                
                # Get batch
                try:
                    batch = next(task_iters[task_name])
                except StopIteration:
                    task_iters[task_name] = iter(task_loaders[task_name])
                    batch = next(task_iters[task_name])
                
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        out = self.model(batch, task_name)
                        preds = out['output'].view(-1)
                        trues = batch.y.view(-1)
                        mse = loss_fn(preds, trues)
                        reg = out.get('losses', torch.tensor(0.0, device=self.device))
                        loss = mse_weight * mse + reg
                    
                    if not torch.isfinite(loss):
                        continue
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = self.model(batch, task_name)
                    preds = out['output'].view(-1)
                    trues = batch.y.view(-1)
                    mse = loss_fn(preds, trues)
                    reg = out.get('losses', torch.tensor(0.0, device=self.device))
                    loss = mse_weight * mse + reg
                    
                    if not torch.isfinite(loss):
                        continue
                    
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()
                
                epoch_loss += float(loss.detach().cpu())
                task_losses[task_name] += float(loss.detach().cpu())
                task_batch_counts[task_name] += 1
            
            # Record history
            train_loss = epoch_loss / max(1, max_batches)
            history['train_loss'].append(train_loss)
            
            for task in self.task_names:
                avg_task_loss = task_losses[task] / max(1, task_batch_counts[task])
                history['task_train_loss'][task].append(avg_task_loss)
            
            # Validation
            metrics = self._evaluate_all_tasks(batch_size=batch_size)
            
            val_spearmans = [
                metrics[task]['valid']['spearman']
                for task in self.task_names
                if not np.isnan(metrics[task]['valid']['spearman'])
            ]
            avg_val_spearman = np.mean(val_spearmans) if val_spearmans else -1.0
            history['avg_val_spearman'].append(avg_val_spearman)
            
            scheduler.step(avg_val_spearman)
            
            # Check for improvement
            if avg_val_spearman > best_avg_val_spearman + 1e-8:
                best_avg_val_spearman = avg_val_spearman
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state': best_state,
                    'optimizer_state': optimizer.state_dict(),
                    'metrics': metrics,
                    'avg_val_spearman': avg_val_spearman
                }, str(self.out_dir / "best_checkpoint.pt"))
            else:
                epochs_no_improve += 1
            
            # Logging
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"\nEpoch {epoch}/{epochs}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Avg Val Spearman: {avg_val_spearman:.4f} (best: {best_avg_val_spearman:.4f})")
                for task in self.task_names:
                    val_sp = metrics[task]['valid']['spearman']
                    val_rmse = metrics[task]['valid']['rmse_denorm']
                    logger.info(f"    {task:25s}: Spearman={val_sp:.4f}, RMSE={val_rmse:.4f}")
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Final evaluation and save predictions
        logger.info("\n" + "=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)
        
        final_metrics = self._evaluate_all_tasks(batch_size=batch_size)
        
        for task_name in self.task_names:
            task_dir = self.out_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            ds = self.task_datasets[task_name]
            
            # Save predictions
            for split, graphs in [('train', ds.train), ('valid', ds.valid), ('test', ds.test)]:
                if len(graphs) > 0:
                    y_true, y_pred = self._predict_list(task_name, graphs, batch_size)
                    np.save(str(task_dir / f"y_{split}.npy"), y_true)
                    np.save(str(task_dir / f"y_pred_{split}.npy"), y_pred)
            
            # Save task results
            task_results = {
                'task_name': task_name,
                'best_epoch': best_epoch,
                'norm_stats': self.norm_stats[task_name],
                'metrics': final_metrics[task_name]
            }
            
            with open(str(task_dir / "results.json"), 'w') as f:
                json.dump(task_results, f, indent=2)
            
            # Log results
            logger.info(f"\n{task_name}:")
            for split in ['train', 'valid', 'test']:
                m = final_metrics[task_name][split]
                logger.info(
                    f"  {split.capitalize():5s} - Spearman: {m['spearman']:.4f}, "
                    f"RMSE: {m['rmse_denorm']:.4f}, Pearson: {m['pearson']:.4f}"
                )
        
        # Save overall results
        overall_results = {
            'best_epoch': best_epoch,
            'best_avg_val_spearman': best_avg_val_spearman,
            'history': history,
            'final_metrics': final_metrics
        }
        
        with open(str(self.out_dir / "overall_results.json"), 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        torch.save(self.model.state_dict(), str(self.out_dir / "final_model.pt"))
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Results saved to: {self.out_dir}")
        logger.info(f"{'=' * 60}")
        
        return overall_results


class SingleTaskFinetuneTrainer:
    """
    Trainer for single-task regression finetuning.
    
    Simpler interface when training on a single task.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_name: str,
        train_graphs: List,
        valid_graphs: List,
        test_graphs: List = None,
        norm_stats: Dict[str, float] = None,
        device: str = None,
        out_dir: Path = None
    ):
        # Create a FinetuneDataset-like object
        class SimpleDataset:
            def __init__(self, train, valid, test):
                self.train = train
                self.valid = valid
                self.test = test or []
        
        task_datasets = {task_name: SimpleDataset(train_graphs, valid_graphs, test_graphs or [])}
        norm_stats_dict = {task_name: norm_stats or {'mean': 0.0, 'std': 1.0}}
        
        self.trainer = MultiTaskFinetuneTrainer(
            model=model,
            task_datasets=task_datasets,
            norm_stats=norm_stats_dict,
            device=device,
            out_dir=out_dir
        )
        self.task_name = task_name
    
    def train(self, **kwargs) -> Dict:
        """Train on the single task."""
        return self.trainer.train(**kwargs)
