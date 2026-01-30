"""
Finetuning trainer for multi-task regression.

Trains on regression tasks using a pretrained encoder,
with support for frozen or unfrozen encoder weights.
"""

import copy
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr

from .datasets import create_data_loaders, collate_with_padding

logger = logging.getLogger(__name__)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation, handling edge cases."""
    if len(y_true) < 2:
        return float('nan')
    try:
        r = spearmanr(y_true, y_pred).correlation
        return -1.0 if np.isnan(r) else float(r)
    except Exception:
        return float('nan')


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation, handling edge cases."""
    if len(y_true) < 2:
        return float('nan')
    try:
        r = pearsonr(y_true, y_pred)[0]
        return float('nan') if np.isnan(r) else float(r)
    except Exception:
        return float('nan')


class FinetuneTrainer:
    """
    Trainer for multi-task regression finetuning.
    
    Supports training on multiple regression tasks simultaneously,
    with proportional or round-robin task sampling.
    
    Args:
        model: MultiTaskRegressionModel instance
        task_datasets: Dict mapping task_name -> {'train': [graphs], 'valid': [...], 'test': [...]}
        device: Device to train on
        out_dir: Directory for checkpoints and logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_datasets: Dict[str, Dict[str, List[Data]]],
        device: Optional[torch.device] = None,
        out_dir: str = "./checkpoints"
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer with {len(self.task_names)} tasks:")
        for task_name in self.task_names:
            train_size = len(task_datasets[task_name]['train'])
            valid_size = len(task_datasets[task_name]['valid'])
            test_size = len(task_datasets[task_name]['test'])
            logger.info(f"  {task_name}: train={train_size}, valid={valid_size}, test={test_size}")
    
    def _predict(
        self,
        task_name: str,
        graphs: List[Data],
        batch_size: int = 128
    ) -> tuple:
        """Get predictions for a list of graphs."""
        self.model.eval()
        
        if len(graphs) == 0:
            return np.array([]), np.array([])
        
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_with_padding)
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
    
    def _evaluate_all_tasks(self, batch_size: int = 128) -> Dict[str, Dict]:
        """Evaluate all tasks on all splits."""
        metrics = {}
        
        for task_name in self.task_names:
            task_metrics = {}
            
            for split in ['train', 'valid', 'test']:
                graphs = self.task_datasets[task_name][split]
                if len(graphs) == 0:
                    task_metrics[split] = {
                        'spearman': float('nan'),
                        'rmse': float('nan'),
                        'pearson': float('nan')
                    }
                    continue
                
                y_true, y_pred = self._predict(task_name, graphs, batch_size)
                
                task_metrics[split] = {
                    'spearman': spearman_correlation(y_true, y_pred),
                    'rmse': math.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else float('nan'),
                    'pearson': pearson_correlation(y_true, y_pred)
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
        task_sampling: str = 'proportional',
        validate_batch_size: Optional[int] = None,
        num_workers: int = 0
    ) -> Dict[str, Any]:
        """
        Run finetuning.
        
        Args:
            epochs: Maximum training epochs
            batch_size: Training batch size
            lr: Learning rate
            weight_decay: Weight decay
            mse_weight: Weight for MSE loss
            grad_clip: Gradient clipping value
            lr_patience: Patience for LR reduction
            early_stop_patience: Patience for early stopping
            task_sampling: 'proportional' or 'round_robin'
            validate_batch_size: Batch size for validation
            num_workers: Data loader workers
            
        Returns:
            Dict with training results and metrics
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        validate_batch_size = validate_batch_size or batch_size
        
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=lr_patience, factor=0.5
        )
        
        use_amp = torch.cuda.is_available()
        scaler = torch.amp.GradScaler() if use_amp else None
        
        # Create data loaders for each task
        task_loaders = {}
        task_iters = {}
        for task_name in self.task_names:
            train_graphs = self.task_datasets[task_name]['train']
            loader = DataLoader(
                train_graphs,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_with_padding
            )
            task_loaders[task_name] = loader
            task_iters[task_name] = iter(loader)
        
        # Task sampling probabilities
        if task_sampling == 'proportional':
            total_samples = sum(len(self.task_datasets[t]['train']) for t in self.task_names)
            task_probs = [
                len(self.task_datasets[t]['train']) / total_samples
                for t in self.task_names
            ]
        else:
            task_probs = [1.0 / len(self.task_names)] * len(self.task_names)
        
        loss_fn = nn.MSELoss()
        
        best_state = None
        best_avg_val_spearman = -999.0
        best_epoch = -1
        epochs_no_improve = 0
        
        history = {
            'train_loss': [],
            'task_train_loss': {task: [] for task in self.task_names},
            'avg_val_spearman': []
        }
        
        logger.info(f"Starting finetuning for {epochs} epochs with {task_sampling} sampling...")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            task_losses = defaultdict(float)
            task_batch_counts = defaultdict(int)
            
            max_batches = max(len(task_loaders[task]) for task in self.task_names)
            
            for batch_idx in range(max_batches):
                # Sample task
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
            
            train_loss = epoch_loss / max(1, max_batches)
            history['train_loss'].append(train_loss)
            
            for task in self.task_names:
                avg_task_loss = task_losses[task] / max(1, task_batch_counts[task])
                history['task_train_loss'][task].append(avg_task_loss)
            
            # Validation
            metrics = self._evaluate_all_tasks(batch_size=validate_batch_size)
            
            val_spearmans = [
                metrics[task]['valid']['spearman']
                for task in self.task_names
                if not np.isnan(metrics[task]['valid']['spearman'])
            ]
            avg_val_spearman = np.mean(val_spearmans) if val_spearmans else -1.0
            history['avg_val_spearman'].append(avg_val_spearman)
            
            scheduler.step(avg_val_spearman)
            
            # Checkpointing
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
                logger.info(f"Epoch {epoch}/{epochs}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Avg Val Spearman: {avg_val_spearman:.4f} (best: {best_avg_val_spearman:.4f})")
                for task in self.task_names:
                    val_sp = metrics[task]['valid']['spearman']
                    val_rmse = metrics[task]['valid']['rmse']
                    logger.info(f"    {task}: Spearman={val_sp:.4f}, RMSE={val_rmse:.4f}")
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Final evaluation and save results
        logger.info("Running final evaluation...")
        final_metrics = self._evaluate_all_tasks(batch_size=validate_batch_size)
        
        self._save_task_results(final_metrics, best_epoch, validate_batch_size)
        
        # Save overall results
        overall_results = {
            'best_epoch': best_epoch,
            'best_avg_val_spearman': float(best_avg_val_spearman),
            'history': history,
            'final_metrics': final_metrics
        }
        
        with open(self.out_dir / "overall_results.json", 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        # Save final model
        torch.save(self.model.state_dict(), str(self.out_dir / "final_model.pt"))
        
        logger.info(f"Finetuning complete! Best epoch: {best_epoch}")
        logger.info(f"Best avg validation Spearman: {best_avg_val_spearman:.4f}")
        
        return overall_results
    
    def _save_task_results(
        self,
        final_metrics: Dict,
        best_epoch: int,
        batch_size: int
    ):
        """Save per-task results and predictions."""
        for task_name in self.task_names:
            task_dir = self.out_dir.parent / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            for split in ['train', 'valid', 'test']:
                graphs = self.task_datasets[task_name][split]
                if len(graphs) > 0:
                    y_true, y_pred = self._predict(task_name, graphs, batch_size)
                    np.save(str(task_dir / f"y_{split}.npy"), y_true)
                    np.save(str(task_dir / f"y_pred_{split}.npy"), y_pred)
            
            # Save metrics
            task_results = {
                'task_name': task_name,
                'best_epoch': best_epoch,
                'metrics': final_metrics[task_name]
            }
            
            with open(task_dir / "results.json", 'w') as f:
                json.dump(task_results, f, indent=2)
            
            # Log final metrics
            logger.info(f"\n{task_name}:")
            for split in ['train', 'valid', 'test']:
                m = final_metrics[task_name][split]
                logger.info(
                    f"  {split.capitalize()}: Spearman={m['spearman']:.4f}, "
                    f"RMSE={m['rmse']:.4f}, Pearson={m['pearson']:.4f}"
                )
