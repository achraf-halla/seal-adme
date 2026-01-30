"""
Pretraining trainer for multi-task classification.

Trains encoder on classification tasks from TDC to learn general
molecular representations before finetuning on regression tasks.
"""

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .datasets import PretrainDataset, BalancedMultiTaskSampler, collate_with_padding

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    Trainer for multi-task classification pretraining.
    
    Uses balanced task sampling to train a shared encoder across
    multiple binary classification tasks.
    
    Args:
        model: MultiTaskPretrainModel instance
        train_dataset: PretrainDataset for training
        valid_dataset: PretrainDataset for validation
        device: Device to train on
        out_dir: Directory for checkpoints and logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: PretrainDataset,
        valid_dataset: PretrainDataset,
        device: Optional[torch.device] = None,
        out_dir: str = "./checkpoints"
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def _predict_task(
        self,
        dataset: PretrainDataset,
        task_name: str,
        split: str,
        batch_size: int = 128
    ) -> tuple:
        """Get predictions for a task/split."""
        self.model.eval()
        indices = dataset.get_task_indices(task_name, split)
        
        if len(indices) == 0:
            return np.array([]), np.array([])
        
        graphs = [dataset.load_graph(i) for i in indices]
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_with_padding)
        
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
        dataset: PretrainDataset,
        split: str
    ) -> Dict[str, Any]:
        """Evaluate all tasks and compute metrics."""
        metrics = {'auroc': {}, 'auprc': {}}
        
        for task in dataset.task_names:
            y_true, y_pred = self._predict_task(dataset, task, split)
            
            if len(y_true) > 0 and len(np.unique(y_true)) > 1:
                try:
                    auroc = roc_auc_score(y_true, y_pred)
                    auprc = average_precision_score(y_true, y_pred)
                except Exception:
                    auroc, auprc = 0.0, 0.0
            else:
                auroc, auprc = 0.0, 0.0
            
            metrics['auroc'][task] = float(auroc)
            metrics['auprc'][task] = float(auprc)
        
        avg_auroc = np.mean(list(metrics['auroc'].values()))
        avg_auprc = np.mean(list(metrics['auprc'].values()))
        
        return {
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc'],
            'avg_auroc': float(avg_auroc),
            'avg_auprc': float(avg_auprc)
        }
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        samples_per_task_per_epoch: Optional[int] = None,
        grad_clip: float = 1.0,
        lr_patience: int = 5,
        early_stop_patience: int = 15
    ) -> Dict[str, Any]:
        """
        Run pretraining.
        
        Args:
            epochs: Maximum training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay
            samples_per_task_per_epoch: Samples per task (None = balanced)
            grad_clip: Gradient clipping value
            lr_patience: Patience for LR reduction
            early_stop_patience: Patience for early stopping
            
        Returns:
            Dict with training results and metrics
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
        
        history = {'train_loss': [], 'val_auroc': [], 'val_auprc': []}
        
        logger.info(f"Starting pretraining for {epochs} epochs...")
        logger.info(f"Tasks: {self.train_dataset.task_names}")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            
            sampler = BalancedMultiTaskSampler(
                self.train_dataset, 'train', batch_size, samples_per_task_per_epoch
            )
            
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_indices in sampler:
                graphs = [self.train_dataset.load_graph(i) for i in batch_indices]
                
                # Group by task
                task_batches = defaultdict(list)
                for g in graphs:
                    task_batches[g.task_name].append(g)
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for task_name, task_graphs in task_batches.items():
                    loader = DataLoader(task_graphs, batch_size=len(task_graphs), shuffle=False, collate_fn=collate_with_padding)
                    batch = next(iter(loader)).to(self.device)
                    
                    out = self.model(batch, task_name)
                    preds = out['output'].view(-1)
                    trues = batch.y.view(-1)
                    
                    task_loss = loss_fn(preds, trues) + out['losses']
                    batch_loss = batch_loss + task_loss
                
                if not torch.isfinite(batch_loss):
                    continue
                
                batch_loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                
                epoch_loss += float(batch_loss.detach().cpu())
                n_batches += 1
            
            train_loss = epoch_loss / max(1, n_batches)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_metrics = self._evaluate_all_tasks(self.valid_dataset, 'valid')
            val_auroc = val_metrics['avg_auroc']
            val_auprc = val_metrics['avg_auprc']
            
            history['val_auroc'].append(val_auroc)
            history['val_auprc'].append(val_auprc)
            
            logger.info(
                f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - "
                f"Val AUROC: {val_auroc:.4f} - Val AUPRC: {val_auprc:.4f}"
            )
            
            scheduler.step(val_auroc)
            
            # Checkpointing
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
                }, str(self.out_dir / "best_checkpoint.pt"))
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        # Save encoder separately for finetuning
        torch.save(
            self.model.encoder.state_dict(),
            str(self.out_dir / "pretrained_encoder.pt")
        )
        
        # Final evaluation
        train_metrics = self._evaluate_all_tasks(self.train_dataset, 'train')
        val_metrics = self._evaluate_all_tasks(self.valid_dataset, 'valid')
        
        results = {
            'best_epoch': best_epoch,
            'best_val_auroc': float(best_val_auroc),
            'history': history,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics
        }
        
        with open(self.out_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Pretraining complete! Best epoch: {best_epoch}, Best AUROC: {best_val_auroc:.4f}")
        logger.info(f"Encoder saved to: {self.out_dir / 'pretrained_encoder.pt'}")
        
        return results
