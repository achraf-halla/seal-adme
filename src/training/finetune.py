"""
Finetuning trainer for regression tasks.

Trains task-specific heads on top of a pretrained encoder.
Supports both frozen and unfrozen encoder training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .datasets import FinetuneDataset, collate_with_padding, create_dataloader

logger = logging.getLogger(__name__)


class FinetuneTrainer:
    """
    Trainer for single-task regression finetuning.
    
    Trains a model on a single regression task using MSE loss.
    Predictions are in normalized space (mean=0, std=1).
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_name: str,
        train_graphs: List,
        valid_graphs: List,
        test_graphs: List = None,
        batch_size: int = 64,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = None,
        checkpoint_dir: Path = None,
        norm_stats: Dict[str, float] = None
    ):
        """
        Args:
            model: MultiTaskRegressionModel
            task_name: Name of the task
            train_graphs: Training graphs
            valid_graphs: Validation graphs
            test_graphs: Test graphs (optional)
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            norm_stats: {'mean': float, 'std': float} for denormalization
        """
        self.model = model
        self.task_name = task_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_stats = norm_stats or {'mean': 0.0, 'std': 1.0}
        
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        
        # Create dataloaders
        self.train_loader = create_dataloader(train_graphs, batch_size, shuffle=True)
        self.valid_loader = create_dataloader(valid_graphs, batch_size, shuffle=False)
        self.test_loader = create_dataloader(test_graphs, batch_size, shuffle=False) if test_graphs else None
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_rmse': [],
            'valid_loss': [],
            'valid_rmse': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_se = 0.0
        n_samples = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            
            out = self.model(batch, self.task_name)
            predictions = out['output']
            reg_loss = out['losses']
            
            labels = batch.y.view(-1, 1)
            
            mse_loss = self.criterion(predictions, labels)
            loss = mse_loss + reg_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(batch)
            total_se += ((predictions - labels) ** 2).sum().item()
            n_samples += len(batch)
        
        avg_loss = total_loss / n_samples
        rmse = np.sqrt(total_se / n_samples)
        
        return avg_loss, rmse
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on a data loader."""
        self.model.eval()
        
        total_loss = 0.0
        total_se = 0.0
        n_samples = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            out = self.model(batch, self.task_name)
            predictions = out['output']
            
            labels = batch.y.view(-1, 1)
            
            mse_loss = self.criterion(predictions, labels)
            
            total_loss += mse_loss.item() * len(batch)
            total_se += ((predictions - labels) ** 2).sum().item()
            n_samples += len(batch)
        
        avg_loss = total_loss / n_samples
        rmse = np.sqrt(total_se / n_samples)
        
        return avg_loss, rmse
    
    def train(
        self,
        epochs: int,
        patience: int = 20,
        log_interval: int = 1
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
            log_interval: How often to log
            
        Returns:
            Training history
        """
        logger.info(f"Starting finetuning for {self.task_name}")
        logger.info(f"Train: {len(self.train_loader.dataset)}, Valid: {len(self.valid_loader.dataset)}")
        logger.info(f"Device: {self.device}")
        
        best_valid_rmse = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_rmse = self.train_epoch()
            
            # Validate
            valid_loss, valid_rmse = self.evaluate(self.valid_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_rmse'].append(valid_rmse)
            
            # Log
            if epoch % log_interval == 0:
                # Denormalize RMSE for interpretability
                denorm_train_rmse = train_rmse * self.norm_stats['std']
                denorm_valid_rmse = valid_rmse * self.norm_stats['std']
                
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train RMSE: {denorm_train_rmse:.4f} | "
                    f"Valid RMSE: {denorm_valid_rmse:.4f}"
                )
            
            # Early stopping
            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                best_epoch = epoch
                patience_counter = 0
                
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, valid_rmse, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if self.checkpoint_dir:
            best_path = self.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                self.load_checkpoint(best_path)
        
        # Test evaluation
        if self.test_loader:
            test_loss, test_rmse = self.evaluate(self.test_loader)
            denorm_test_rmse = test_rmse * self.norm_stats['std']
            self.history['test_rmse'] = test_rmse
            self.history['test_rmse_denorm'] = denorm_test_rmse
            logger.info(f"Test RMSE: {denorm_test_rmse:.4f}")
        
        self.history['best_epoch'] = best_epoch
        self.history['best_valid_rmse'] = best_valid_rmse
        
        # Save history
        if self.checkpoint_dir:
            with open(self.checkpoint_dir / "history.json", 'w') as f:
                json.dump(self.history, f, indent=2)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, rmse: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rmse': rmse,
            'task_name': self.task_name,
            'norm_stats': self.norm_stats
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    @torch.no_grad()
    def predict(self, graphs: List, return_denorm: bool = True) -> np.ndarray:
        """
        Make predictions on a list of graphs.
        
        Args:
            graphs: List of PyG Data objects
            return_denorm: Whether to denormalize predictions
            
        Returns:
            Numpy array of predictions
        """
        self.model.eval()
        loader = create_dataloader(graphs, self.batch_size, shuffle=False)
        
        all_preds = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch, self.task_name)
            preds = out['output'].cpu().numpy()
            all_preds.append(preds)
        
        preds = np.vstack(all_preds).flatten()
        
        if return_denorm:
            preds = preds * self.norm_stats['std'] + self.norm_stats['mean']
        
        return preds


def train_all_finetune_tasks(
    model_builder,
    finetune_datasets: Dict[str, FinetuneDataset],
    norm_stats: Dict[str, Dict[str, float]],
    encoder_path: Path = None,
    output_dir: Path = None,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 20,
    device: str = None
) -> Dict[str, Dict]:
    """
    Train models for all finetuning tasks.
    
    Args:
        model_builder: Function to build model (task_names, encoder_path) -> model
        finetune_datasets: Dict of task_name -> FinetuneDataset
        norm_stats: Dict of task_name -> {'mean': float, 'std': float}
        encoder_path: Path to pretrained encoder
        output_dir: Base output directory
        epochs: Max epochs per task
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        device: Device
        
    Returns:
        Dict of task_name -> training history
    """
    results = {}
    
    for task_name, dataset in finetune_datasets.items():
        logger.info("=" * 60)
        logger.info(f"Training: {task_name}")
        logger.info("=" * 60)
        
        # Build model for this task
        model = model_builder(
            task_names=[task_name],
            encoder_checkpoint=str(encoder_path) if encoder_path else None
        )
        
        # Get checkpoint dir
        task_output = output_dir / task_name if output_dir else None
        
        # Create trainer
        trainer = FinetuneTrainer(
            model=model,
            task_name=task_name,
            train_graphs=dataset.train,
            valid_graphs=dataset.valid,
            test_graphs=dataset.test,
            batch_size=batch_size,
            lr=lr,
            device=device,
            checkpoint_dir=task_output,
            norm_stats=norm_stats.get(task_name, {'mean': 0.0, 'std': 1.0})
        )
        
        # Train
        history = trainer.train(epochs=epochs, patience=patience)
        results[task_name] = history
    
    return results
