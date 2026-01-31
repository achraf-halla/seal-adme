"""
Pretraining trainer for multi-task classification.

Trains a fragment-aware encoder on multiple classification tasks
with balanced sampling to ensure equal representation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .datasets import (
    PretrainDataset,
    BalancedMultiTaskSampler,
    collate_with_padding,
)

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    Trainer for multi-task pretraining on classification tasks.
    
    Uses balanced sampling and binary cross-entropy loss.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: PretrainDataset,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = None,
        checkpoint_dir: Path = None
    ):
        """
        Args:
            model: MultiTaskPretrainModel
            dataset: PretrainDataset with loaded graphs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sampler
        self.sampler = BalancedMultiTaskSampler(
            dataset=dataset,
            batch_size=batch_size
        )
        
        self.history = {
            'train_loss': [],
            'task_losses': {task: [] for task in dataset.task_names},
            'task_accs': {task: [] for task in dataset.task_names}
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        task_losses = {task: 0.0 for task in self.dataset.task_names}
        task_correct = {task: 0 for task in self.dataset.task_names}
        task_total = {task: 0 for task in self.dataset.task_names}
        n_batches = 0
        
        for batch_indices in tqdm(self.sampler, desc="Training", leave=False):
            # Get graphs for this batch
            graphs = [self.dataset[i] for i in batch_indices]
            
            # Group by task
            task_graphs = {}
            for g in graphs:
                task = g.task_name
                if task not in task_graphs:
                    task_graphs[task] = []
                task_graphs[task].append(g)
            
            batch_loss = 0.0
            
            # Forward pass for each task in batch
            for task_name, task_batch in task_graphs.items():
                if not task_batch:
                    continue
                
                # Collate and move to device
                batch = collate_with_padding(task_batch).to(self.device)
                
                # Forward
                out = self.model(batch, task_name)
                predictions = out['output']
                reg_loss = out['losses']
                
                # Get labels
                labels = batch.y.view(-1, 1).to(self.device)
                
                # Loss
                task_loss = self.criterion(predictions, labels)
                loss = task_loss + reg_loss
                
                batch_loss += loss
                task_losses[task_name] += task_loss.item()
                
                # Accuracy
                preds_binary = (torch.sigmoid(predictions) > 0.5).float()
                correct = (preds_binary == labels).sum().item()
                task_correct[task_name] += correct
                task_total[task_name] += len(task_batch)
            
            # Backward
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += batch_loss.item()
            n_batches += 1
        
        # Compute averages
        avg_loss = epoch_loss / max(n_batches, 1)
        
        metrics = {'loss': avg_loss}
        for task in self.dataset.task_names:
            if task_total[task] > 0:
                metrics[f'{task}_loss'] = task_losses[task] / max(n_batches, 1)
                metrics[f'{task}_acc'] = task_correct[task] / task_total[task]
        
        return metrics
    
    def train(
        self,
        epochs: int,
        log_interval: int = 1,
        save_interval: int = 10
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs
            log_interval: How often to log metrics
            save_interval: How often to save checkpoints
            
        Returns:
            Training history
        """
        logger.info(f"Starting pretraining for {epochs} epochs")
        logger.info(f"Tasks: {self.dataset.task_names}")
        logger.info(f"Device: {self.device}")
        
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch()
            
            self.history['train_loss'].append(metrics['loss'])
            for task in self.dataset.task_names:
                if f'{task}_loss' in metrics:
                    self.history['task_losses'][task].append(metrics[f'{task}_loss'])
                if f'{task}_acc' in metrics:
                    self.history['task_accs'][task].append(metrics[f'{task}_acc'])
            
            # Log
            if epoch % log_interval == 0:
                avg_acc = sum(
                    metrics.get(f'{t}_acc', 0) for t in self.dataset.task_names
                ) / len(self.dataset.task_names)
                
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Avg Acc: {avg_acc:.4f}"
                )
            
            # Save checkpoint
            if self.checkpoint_dir and epoch % save_interval == 0:
                self.save_checkpoint(epoch, metrics['loss'])
            
            # Save best
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, metrics['loss'], is_best=True)
        
        # Save final encoder
        if self.checkpoint_dir:
            self.save_encoder()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'task_names': self.dataset.task_names
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def save_encoder(self):
        """Save just the encoder for finetuning."""
        path = self.checkpoint_dir / "pretrained_encoder.pt"
        torch.save(self.model.encoder.state_dict(), path)
        logger.info(f"Saved encoder to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint['epoch']
