"""
Training loops for SEAL models.

Provides trainers for both pretraining (classification) and fine-tuning
(regression) with multitask learning support.
"""

import copy
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr

from .datasets import (
    MultiTaskDataset,
    TaskGraphDataset,
    BalancedMultiTaskSampler,
    collate_by_task,
)

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base class for SEAL trainers."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        output_dir: Path = None
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() 
            else torch.device("cpu")
        )
        self.model = model.to(self.device)
        
        self.output_dir = Path(output_dir) if output_dir else Path("./checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _predict(
        self,
        task_name: str,
        graphs: List,
        batch_size: int = 128
    ) -> tuple:
        """Make predictions on a list of graphs."""
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
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        filename: str = "checkpoint.pt"
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'encoder_state': self.model.encoder.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_encoder(self, filename: str = "pretrained_encoder.pt"):
        """Save just the encoder weights."""
        torch.save(
            self.model.encoder.state_dict(),
            self.output_dir / filename
        )


class PretrainTrainer(BaseTrainer):
    """
    Trainer for pretraining on classification tasks.
    
    Uses balanced sampling across tasks and BCE loss.
    
    Args:
        model: PretrainModel instance
        train_dataset: MultiTaskDataset for training
        valid_dataset: MultiTaskDataset for validation
        device: Torch device
        output_dir: Directory for checkpoints and results
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: MultiTaskDataset,
        valid_dataset: MultiTaskDataset,
        device: torch.device = None,
        output_dir: Path = None
    ):
        super().__init__(model, device, output_dir)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
    
    def _evaluate_classification(
        self,
        dataset: MultiTaskDataset,
        split: str
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate classification metrics on all tasks."""
        metrics = {'auroc': {}, 'auprc': {}}
        
        for task in dataset.task_names:
            indices = dataset.get_task_indices(task, split)
            if len(indices) == 0:
                continue
            
            graphs = [dataset.load_graph(i) for i in indices]
            y_true, y_pred = self._predict(task, graphs)
            
            # Apply sigmoid for probabilities
            y_pred_prob = 1 / (1 + np.exp(-y_pred))
            
            if len(np.unique(y_true)) > 1:
                try:
                    auroc = roc_auc_score(y_true, y_pred_prob)
                    auprc = average_precision_score(y_true, y_pred_prob)
                except Exception:
                    auroc, auprc = 0.0, 0.0
            else:
                auroc, auprc = 0.0, 0.0
            
            metrics['auroc'][task] = float(auroc)
            metrics['auprc'][task] = float(auprc)
        
        metrics['avg_auroc'] = float(np.mean(list(metrics['auroc'].values())))
        metrics['avg_auprc'] = float(np.mean(list(metrics['auprc'].values())))
        
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
    ) -> Dict[str, Any]:
        """
        Run pretraining loop.
        
        Args:
            epochs: Maximum number of epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            samples_per_task: Samples per task per epoch
            grad_clip: Gradient clipping threshold
            lr_patience: Epochs before LR reduction
            early_stop_patience: Epochs before early stopping
            
        Returns:
            Training results dictionary
        """
        optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=lr_patience, factor=0.5
        )
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        best_state = None
        best_val_auroc = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        
        history = {'train_loss': [], 'val_auroc': [], 'val_auprc': []}
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            
            sampler = BalancedMultiTaskSampler(
                self.train_dataset,
                split='train',
                batch_size=batch_size,
                samples_per_task=samples_per_task
            )
            
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_indices in sampler:
                graphs = [self.train_dataset.load_graph(i) for i in batch_indices]
                task_batches = collate_by_task(graphs)
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for task_name, task_graphs in task_batches.items():
                    loader = DataLoader(task_graphs, batch_size=len(task_graphs))
                    batch = next(iter(loader)).to(self.device)
                    
                    out = self.model(batch, task_name)
                    preds = out['output'].view(-1)
                    trues = batch.y.view(-1)
                    
                    task_loss = loss_fn(preds, trues) + out['losses']
                    batch_loss += task_loss
                
                if not torch.isfinite(batch_loss):
                    continue
                
                batch_loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    )
                optimizer.step()
                
                epoch_loss += float(batch_loss.detach().cpu())
                n_batches += 1
            
            train_loss = epoch_loss / max(1, n_batches)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_metrics = self._evaluate_classification(
                self.valid_dataset, 'valid'
            )
            val_auroc = val_metrics['avg_auroc']
            
            history['val_auroc'].append(val_auroc)
            history['val_auprc'].append(val_metrics['avg_auprc'])
            
            logger.info(
                f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - "
                f"Val AUROC: {val_auroc:.4f}"
            )
            
            scheduler.step(val_auroc)
            
            # Check for improvement
            if val_auroc > best_val_auroc + 1e-6:
                best_val_auroc = val_auroc
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                
                self.save_checkpoint(epoch, val_metrics, "best_checkpoint.pt")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        # Save encoder
        self.save_encoder()
        
        # Final evaluation
        train_metrics = self._evaluate_classification(self.train_dataset, 'train')
        val_metrics = self._evaluate_classification(self.valid_dataset, 'valid')
        
        results = {
            'best_epoch': best_epoch,
            'best_val_auroc': float(best_val_auroc),
            'history': history,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics
        }
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


class RegressionTrainer(BaseTrainer):
    """
    Trainer for fine-tuning on regression tasks.
    
    Uses MSE loss with support for multiple tasks and various
    sampling strategies.
    
    Args:
        model: RegressionModel instance
        task_datasets: Dict mapping task names to TaskGraphDataset
        device: Torch device
        output_dir: Directory for checkpoints and results
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_datasets: Dict[str, TaskGraphDataset],
        device: torch.device = None,
        output_dir: Path = None
    ):
        super().__init__(model, device, output_dir)
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        
        logger.info(f"Initialized trainer with {len(self.task_names)} tasks:")
        for task_name in self.task_names:
            ds = task_datasets[task_name]
            logger.info(
                f"  {task_name}: train={len(ds.train)}, "
                f"valid={len(ds.valid)}, test={len(ds.test)}"
            )
    
    def _evaluate_regression(
        self,
        batch_size: int = 128
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate regression metrics on all tasks and splits."""
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
                
                # Compute metrics
                sp, _ = spearmanr(y_true, y_pred)
                spearman = float(sp) if not np.isnan(sp) else -1.0
                rmse = math.sqrt(mean_squared_error(y_true, y_pred))
                
                task_metrics[split] = {
                    'spearman': spearman,
                    'rmse': float(rmse)
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
    ) -> Dict[str, Any]:
        """
        Run fine-tuning loop.
        
        Args:
            epochs: Maximum epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            mse_weight: Weight for MSE loss
            grad_clip: Gradient clipping
            lr_patience: Epochs before LR reduction
            early_stop_patience: Epochs before early stopping
            task_sampling: 'round_robin' or 'proportional'
            num_workers: DataLoader workers
            
        Returns:
            Training results dictionary
        """
        optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=lr_patience, factor=0.5
        )
        
        # AMP for faster training on GPU
        use_amp = self.device.type == 'cuda'
        scaler = torch.amp.GradScaler() if use_amp else None
        
        loss_fn = nn.MSELoss()
        
        # Create data loaders
        task_loaders = {}
        task_iters = {}
        for task_name in self.task_names:
            train_graphs = self.task_datasets[task_name].train
            loader = DataLoader(
                train_graphs,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=use_amp
            )
            task_loaders[task_name] = loader
            task_iters[task_name] = iter(loader)
        
        # Task sampling probabilities
        if task_sampling == 'proportional':
            total = sum(len(self.task_datasets[t].train) for t in self.task_names)
            task_probs = [
                len(self.task_datasets[t].train) / total
                for t in self.task_names
            ]
        else:
            task_probs = [1.0 / len(self.task_names)] * len(self.task_names)
        
        best_state = None
        best_val_spearman = -999.0
        best_epoch = -1
        epochs_no_improve = 0
        
        history = {
            'train_loss': [],
            'task_losses': {t: [] for t in self.task_names},
            'avg_val_spearman': []
        }
        
        logger.info(f"Starting training with {task_sampling} sampling")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            
            epoch_loss = 0.0
            task_losses = defaultdict(float)
            task_counts = defaultdict(int)
            
            max_batches = max(len(task_loaders[t]) for t in self.task_names)
            
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
                
                # Forward pass (with AMP if available)
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        out = self.model(batch, task_name)
                        preds = out['output'].view(-1)
                        trues = batch.y.view(-1)
                        loss = mse_weight * loss_fn(preds, trues) + out['losses']
                    
                    if not torch.isfinite(loss):
                        continue
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = self.model(batch, task_name)
                    preds = out['output'].view(-1)
                    trues = batch.y.view(-1)
                    loss = mse_weight * loss_fn(preds, trues) + out['losses']
                    
                    if not torch.isfinite(loss):
                        continue
                    
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    optimizer.step()
                
                epoch_loss += float(loss.detach().cpu())
                task_losses[task_name] += float(loss.detach().cpu())
                task_counts[task_name] += 1
            
            # Record history
            train_loss = epoch_loss / max(1, max_batches)
            history['train_loss'].append(train_loss)
            
            for task in self.task_names:
                avg = task_losses[task] / max(1, task_counts[task])
                history['task_losses'][task].append(avg)
            
            # Validation
            metrics = self._evaluate_regression()
            
            val_spearmans = [
                metrics[t]['valid']['spearman']
                for t in self.task_names
                if not np.isnan(metrics[t]['valid']['spearman'])
            ]
            avg_val_spearman = np.mean(val_spearmans) if val_spearmans else -1.0
            history['avg_val_spearman'].append(avg_val_spearman)
            
            scheduler.step(avg_val_spearman)
            
            # Check for improvement
            if avg_val_spearman > best_val_spearman + 1e-8:
                best_val_spearman = avg_val_spearman
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                
                self.save_checkpoint(epoch, metrics, "best_checkpoint.pt")
            else:
                epochs_no_improve += 1
            
            # Logging
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - "
                    f"Val Spearman: {avg_val_spearman:.4f}"
                )
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        # Final evaluation
        final_metrics = self._evaluate_regression()
        
        # Save predictions
        self._save_predictions(final_metrics)
        
        results = {
            'best_epoch': best_epoch,
            'best_avg_val_spearman': float(best_val_spearman),
            'history': history,
            'final_metrics': final_metrics
        }
        
        with open(self.output_dir / "overall_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        torch.save(self.model.state_dict(), self.output_dir / "final_model.pt")
        
        return results
    
    def _save_predictions(self, metrics: Dict):
        """Save predictions and metrics for each task."""
        for task_name in self.task_names:
            task_dir = self.output_dir.parent / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            for split in ['train', 'valid', 'test']:
                graphs = self.task_datasets[task_name][split]
                if len(graphs) > 0:
                    y_true, y_pred = self._predict(task_name, graphs)
                    np.save(task_dir / f"y_{split}.npy", y_true)
                    np.save(task_dir / f"y_pred_{split}.npy", y_pred)
            
            # Save task results
            task_results = {
                'task_name': task_name,
                'metrics': metrics[task_name]
            }
            
            with open(task_dir / "results.json", 'w') as f:
                json.dump(task_results, f, indent=2)
