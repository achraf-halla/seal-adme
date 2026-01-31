"""
Pretraining trainer for multi-task classification.

Trains a fragment-aware encoder on multiple classification tasks
with balanced sampling to ensure equal representation.

Features:
- Balanced multi-task sampling
- AUROC/AUPRC evaluation
- LR scheduler with patience
- Early stopping
- Gradient clipping
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from .datasets import PretrainDataset, collate_with_padding

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    Trainer for multi-task pretraining on classification tasks.
    
    Uses balanced sampling across tasks and evaluates with AUROC/AUPRC.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: PretrainDataset,
        valid_dataset: PretrainDataset = None,
        device: str = None,
        out_dir: Path = None
    ):
        """
        Args:
            model: MultiTaskPretrainModel
            train_dataset: PretrainDataset for training
            valid_dataset: PretrainDataset for validation (optional, can use train)
            device: Device to train on
            out_dir: Directory for saving checkpoints
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset or train_dataset
        
        self.out_dir = Path(out_dir) if out_dir else Path("checkpoints/pretrain")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.task_names = train_dataset.task_names
        logger.info(f"Initialized PretrainTrainer with {len(self.task_names)} tasks")
        for task in self.task_names:
            n_train = len(train_dataset.get_task_graphs(task))
            logger.info(f"  {task}: {n_train} training samples")
    
    def _predict_task(
        self,
        dataset: PretrainDataset,
        task_name: str,
        batch_size: int = 128
    ) -> tuple:
        """Predict on all samples for a task."""
        self.model.eval()
        graphs = dataset.get_task_graphs(task_name)
        
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
                preds = torch.sigmoid(out['output']).view(-1).cpu().numpy()
                trues = batch.y.view(-1).cpu().numpy()
                preds_list.append(preds)
                trues_list.append(trues)
        
        return np.concatenate(trues_list), np.concatenate(preds_list)
    
    def _evaluate_all_tasks(self, dataset: PretrainDataset) -> Dict:
        """Evaluate AUROC/AUPRC on all tasks."""
        metrics = {'auroc': {}, 'auprc': {}}
        
        for task in self.task_names:
            y_true, y_pred = self._predict_task(dataset, task)
            
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
        
        metrics['avg_auroc'] = float(np.mean(list(metrics['auroc'].values())))
        metrics['avg_auprc'] = float(np.mean(list(metrics['auprc'].values())))
        
        return metrics
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        samples_per_task_per_epoch: int = None,
        grad_clip: float = 1.0,
        lr_patience: int = 5,
        early_stop_patience: int = 15
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            epochs: Maximum number of epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            samples_per_task_per_epoch: Samples per task (None = min task size)
            grad_clip: Gradient clipping norm
            lr_patience: Patience for LR scheduler
            early_stop_patience: Patience for early stopping
            
        Returns:
            Training results dict
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
        
        # Determine samples per task
        if samples_per_task_per_epoch is None:
            samples_per_task_per_epoch = min(
                len(self.train_dataset.get_task_graphs(t)) for t in self.task_names
            )
        
        logger.info(f"Starting pretraining for {epochs} epochs")
        logger.info(f"Samples per task per epoch: {samples_per_task_per_epoch}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            
            # Create balanced task pools
            task_pools = {}
            for task in self.task_names:
                graphs = self.train_dataset.get_task_graphs(task)
                indices = np.random.permutation(len(graphs)).tolist()
                # Repeat if needed
                while len(indices) < samples_per_task_per_epoch:
                    indices.extend(np.random.permutation(len(graphs)).tolist())
                task_pools[task] = [graphs[i] for i in indices[:samples_per_task_per_epoch]]
            
            # Interleave samples from all tasks
            all_graphs = []
            for i in range(samples_per_task_per_epoch):
                for task in self.task_names:
                    all_graphs.append((task, task_pools[task][i]))
            
            np.random.shuffle(all_graphs)
            
            # Group into batches
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(all_graphs), batch_size):
                batch_items = all_graphs[i:i + batch_size]
                
                # Group by task
                task_batches = defaultdict(list)
                for task, graph in batch_items:
                    task_batches[task].append(graph)
                
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device)
                
                for task_name, task_graphs in task_batches.items():
                    batch = collate_with_padding(task_graphs).to(self.device)
                    
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
            val_metrics = self._evaluate_all_tasks(self.valid_dataset)
            val_auroc = val_metrics['avg_auroc']
            val_auprc = val_metrics['avg_auprc']
            
            history['val_auroc'].append(val_auroc)
            history['val_auprc'].append(val_auprc)
            
            logger.info(
                f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}"
            )
            
            scheduler.step(val_auroc)
            
            # Check for improvement
            if val_auroc > best_val_auroc + 1e-6:
                best_val_auroc = val_auroc
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Save best checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state': best_state,
                    'encoder_state': self.model.encoder.state_dict(),
                    'val_auroc': val_auroc,
                    'val_auprc': val_auprc
                }, str(self.out_dir / "best_checkpoint.pt"))
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        # Save pretrained encoder
        torch.save(
            self.model.encoder.state_dict(),
            str(self.out_dir / "pretrained_encoder.pt")
        )
        
        # Final evaluation
        train_metrics = self._evaluate_all_tasks(self.train_dataset)
        val_metrics = self._evaluate_all_tasks(self.valid_dataset)
        
        results = {
            'best_epoch': best_epoch,
            'best_val_auroc': float(best_val_auroc),
            'history': history,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics
        }
        
        with open(self.out_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training complete. Best epoch: {best_epoch}, Best AUROC: {best_val_auroc:.4f}")
        logger.info(f"Encoder saved to: {self.out_dir / 'pretrained_encoder.pt'}")
        
        return results
