"""
Training loops for SEAL multi-task models.

This module provides trainer classes for both pretraining (classification)
and finetuning (regression) phases of SEAL model training.
"""

import copy
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from .datasets import (
    BalancedMultiTaskSampler,
    MultiTaskDataset,
    TaskGraphDataset,
    create_data_loader,
)
from .metrics import (
    MetricTracker,
    compute_classification_metrics,
    compute_regression_metrics,
    aggregate_task_metrics,
)

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    Trainer for multi-task classification pretraining.
    
    Trains a SEAL model on multiple binary classification tasks
    with balanced sampling across tasks.
    
    Args:
        model: MultiTaskModel instance
        train_dataset: MultiTaskDataset for training
        valid_dataset: MultiTaskDataset for validation
        device: Torch device (None = auto-detect)
        output_dir: Directory for saving checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: MultiTaskDataset,
        valid_dataset: MultiTaskDataset,
        device: Optional[torch.device] = None,
        output_dir: Union[str, Path] = "./checkpoints"
    ):
        self.device = device or self._get_device()
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _predict_task(
        self,
        dataset: MultiTaskDataset,
        task_name: str,
        split: str,
        batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for a specific task and split."""
        self.model.eval()
        indices = dataset.get_task_indices(task_name, split)
        
        if len(indices) == 0:
            return np.array([]), np.array([])
        
        graphs = [dataset.load_graph(i) for i in indices]
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        
        preds_list, trues_list = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch, task_name)
                preds = torch.sigmoid(out['output']).view(-1).cpu().numpy()
                trues = batch.y.view(-1).cpu().numpy()
                preds_list.append(preds)
                trues_list.append(trues)
        
        return np.concatenate(trues_list), np.concatenate(preds_list)
    
    def _evaluate_all_tasks(
        self,
        dataset: MultiTaskDataset,
        split: str
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all tasks on a given split."""
        metrics = {}
        for task in dataset.task_names:
            y_true, y_pred = self._predict_task(dataset, task, split)
            metrics[task] = compute_classification_metrics(y_true, y_pred)
        return metrics
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        samples_per_task: Optional[int] = None,
        grad_clip: float = 1.0,
        lr_patience: int = 5,
        early_stop_patience: int = 15
    ) -> Dict:
        """
        Run pretraining loop.
        
        Args:
            epochs: Maximum number of epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            samples_per_task: Samples per task per epoch
            grad_clip: Gradient clipping value
            lr_patience: Patience for LR scheduler
            early_stop_patience: Patience for early stopping
            
        Returns:
            Training results dictionary
        """
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=lr_patience, factor=0.5
        )
        loss_fn = nn.BCEWithLogitsLoss()
        
        best_state = None
        best_val_auroc = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        
        tracker = MetricTracker(['train_loss', 'val_auroc', 'val_auprc'])
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            
            sampler = BalancedMultiTaskSampler(
                self.train_dataset, 'train', batch_size, samples_per_task
            )
            
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_indices in sampler:
                graphs = [self.train_dataset.load_graph(i) for i in batch_indices]
                
                task_batches = defaultdict(list)
                for g in graphs:
                    task_batches[g.task_name].append(g)
                
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device)
                
                for task_name, task_graphs in task_batches.items():
                    loader = DataLoader(task_graphs, batch_size=len(task_graphs))
                    batch = next(iter(loader)).to(self.device)
                    
                    out = self.model(batch, task_name)
                    preds = out['output'].view(-1)
                    trues = batch.y.view(-1)
                    
                    task_loss = loss_fn(preds, trues) + out['losses']
                    batch_loss = batch_loss + task_loss
                
                if torch.isfinite(batch_loss):
                    batch_loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                    n_batches += 1
            
            train_loss = epoch_loss / max(1, n_batches)
            
            val_metrics = self._evaluate_all_tasks(self.valid_dataset, 'valid')
            val_auroc = aggregate_task_metrics(val_metrics, 'auroc')
            val_auprc = aggregate_task_metrics(val_metrics, 'auprc')
            
            tracker.update({
                'train_loss': train_loss,
                'val_auroc': val_auroc,
                'val_auprc': val_auprc
            }, epoch)
            
            logger.info(
                f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - "
                f"Val AUROC: {val_auroc:.4f} - Val AUPRC: {val_auprc:.4f}"
            )
            
            scheduler.step(val_auroc)
            
            if val_auroc > best_val_auroc + 1e-6:
                best_val_auroc = val_auroc
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state': best_state,
                    'encoder_state': self.model.encoder.state_dict(),
                    'val_auroc': val_auroc
                }, self.output_dir / "best_checkpoint.pt")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        torch.save(
            self.model.encoder.state_dict(),
            self.output_dir / "pretrained_encoder.pt"
        )
        
        results = {
            'best_epoch': best_epoch,
            'best_val_auroc': float(best_val_auroc),
            'history': tracker.to_dict(),
            'final_train_metrics': self._evaluate_all_tasks(self.train_dataset, 'train'),
            'final_val_metrics': self._evaluate_all_tasks(self.valid_dataset, 'valid')
        }
        
        with open(self.output_dir / "pretrain_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


class FinetuneTrainer:
    """
    Trainer for multi-task regression finetuning.
    
    Finetunes a SEAL model on multiple regression tasks using
    a pretrained encoder.
    
    Args:
        model: MultiTaskModel instance (with pretrained encoder)
        task_datasets: Dictionary mapping task names to TaskGraphDataset
        device: Torch device (None = auto-detect)
        output_dir: Directory for saving checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_datasets: Dict[str, TaskGraphDataset],
        device: Optional[torch.device] = None,
        output_dir: Union[str, Path] = "./checkpoints"
    ):
        self.device = device or self._get_device()
        self.model = model.to(self.device)
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer with {len(self.task_names)} tasks:")
        for name in self.task_names:
            ds = task_datasets[name]
            logger.info(
                f"  {name}: train={len(ds.train)}, "
                f"valid={len(ds.valid)}, test={len(ds.test)}"
            )
    
    @staticmethod
    def _get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _predict(
        self,
        task_name: str,
        graphs: List,
        batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for a list of graphs."""
        self.model.eval()
        
        if len(graphs) == 0:
            return np.array([]), np.array([])
        
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
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
    
    def _evaluate_all_tasks(
        self,
        batch_size: int = 128
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate all tasks on all splits."""
        metrics = {}
        for task_name in self.task_names:
            task_metrics = {}
            for split in ['train', 'valid', 'test']:
                graphs = self.task_datasets[task_name][split]
                if len(graphs) == 0:
                    task_metrics[split] = {
                        'spearman': float('nan'),
                        'rmse': float('nan')
                    }
                    continue
                
                y_true, y_pred = self._predict(task_name, graphs, batch_size)
                task_metrics[split] = compute_regression_metrics(y_true, y_pred)
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
        Run finetuning loop.
        
        Args:
            epochs: Maximum number of epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            mse_weight: Weight for MSE loss
            grad_clip: Gradient clipping value
            lr_patience: Patience for LR scheduler
            early_stop_patience: Patience for early stopping
            task_sampling: 'round_robin' or 'proportional'
            num_workers: DataLoader workers
            
        Returns:
            Training results dictionary
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=lr_patience, factor=0.5
        )
        
        use_amp = torch.cuda.is_available()
        scaler = torch.amp.GradScaler() if use_amp else None
        
        task_loaders = {}
        task_iters = {}
        for task_name in self.task_names:
            train_graphs = self.task_datasets[task_name].train
            loader = create_data_loader(
                train_graphs, batch_size, shuffle=True, num_workers=num_workers
            )
            task_loaders[task_name] = loader
            task_iters[task_name] = iter(loader)
        
        if task_sampling == 'proportional':
            total = sum(len(self.task_datasets[t].train) for t in self.task_names)
            task_probs = [len(self.task_datasets[t].train) / total for t in self.task_names]
        else:
            task_probs = [1.0 / len(self.task_names)] * len(self.task_names)
        
        best_state = None
        best_val_spearman = -999.0
        best_epoch = -1
        epochs_no_improve = 0
        
        loss_fn = nn.MSELoss()
        tracker = MetricTracker(['train_loss', 'avg_val_spearman'])
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            task_losses = defaultdict(float)
            task_counts = defaultdict(int)
            
            max_batches = max(len(task_loaders[t]) for t in self.task_names)
            
            for batch_idx in range(max_batches):
                if task_sampling == 'round_robin':
                    task_name = self.task_names[batch_idx % len(self.task_names)]
                else:
                    task_name = np.random.choice(self.task_names, p=task_probs)
                
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
                    
                    if torch.isfinite(loss):
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if grad_clip:
                            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    out = self.model(batch, task_name)
                    preds = out['output'].view(-1)
                    trues = batch.y.view(-1)
                    mse = loss_fn(preds, trues)
                    reg = out.get('losses', torch.tensor(0.0, device=self.device))
                    loss = mse_weight * mse + reg
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        if grad_clip:
                            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        optimizer.step()
                
                if torch.isfinite(loss):
                    epoch_loss += loss.item()
                    task_losses[task_name] += loss.item()
                    task_counts[task_name] += 1
            
            train_loss = epoch_loss / max(1, max_batches)
            
            metrics = self._evaluate_all_tasks(batch_size)
            val_spearmans = [
                metrics[t]['valid']['spearman']
                for t in self.task_names
                if not np.isnan(metrics[t]['valid']['spearman'])
            ]
            avg_val_spearman = np.mean(val_spearmans) if val_spearmans else -1.0
            
            tracker.update({
                'train_loss': train_loss,
                'avg_val_spearman': avg_val_spearman
            }, epoch)
            
            scheduler.step(avg_val_spearman)
            
            if avg_val_spearman > best_val_spearman + 1e-8:
                best_val_spearman = avg_val_spearman
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state': best_state,
                    'optimizer_state': optimizer.state_dict(),
                    'metrics': metrics,
                    'avg_val_spearman': avg_val_spearman
                }, self.output_dir / "best_checkpoint.pt")
            else:
                epochs_no_improve += 1
            
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}/{epochs}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(
                    f"  Avg Val Spearman: {avg_val_spearman:.4f} "
                    f"(best: {best_val_spearman:.4f})"
                )
                for t in self.task_names:
                    sp = metrics[t]['valid']['spearman']
                    rmse = metrics[t]['valid']['rmse']
                    logger.info(f"    {t}: Spearman={sp:.4f}, RMSE={rmse:.4f}")
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        final_metrics = self._evaluate_all_tasks(batch_size)
        
        for task_name in self.task_names:
            task_dir = self.output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            for split in ['train', 'valid', 'test']:
                graphs = self.task_datasets[task_name][split]
                if len(graphs) > 0:
                    y_true, y_pred = self._predict(task_name, graphs, batch_size)
                    np.save(task_dir / f"y_{split}.npy", y_true)
                    np.save(task_dir / f"y_pred_{split}.npy", y_pred)
            
            task_results = {
                'task_name': task_name,
                'best_epoch': best_epoch,
                'metrics': final_metrics[task_name]
            }
            with open(task_dir / "results.json", 'w') as f:
                json.dump(task_results, f, indent=2)
        
        results = {
            'best_epoch': best_epoch,
            'best_avg_val_spearman': float(best_val_spearman),
            'history': tracker.to_dict(),
            'final_metrics': final_metrics
        }
        
        with open(self.output_dir / "finetune_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        torch.save(self.model.state_dict(), self.output_dir / "final_model.pt")
        
        logger.info("Training complete")
        for t in self.task_names:
            m = final_metrics[t]['test']
            logger.info(
                f"  {t} (test): Spearman={m['spearman']:.4f}, "
                f"RMSE={m['rmse']:.4f}, Pearson={m['pearson']:.4f}"
            )
        
        return results
