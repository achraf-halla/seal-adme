"""
Task heads and multi-task model wrappers for SEAL.

This module provides task-specific prediction heads and multi-task
model architectures that combine encoders with task heads.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from typing import Dict, List, Optional, Union

from .encoders import BaseEncoder, create_encoder


class TaskHead(nn.Module):
    """
    Simple linear head for a single task.
    
    Maps fragment embeddings to a scalar prediction per fragment,
    then sums to get the molecular prediction.
    
    Args:
        hidden_dim: Dimension of input fragment embeddings
        output_dim: Output dimension (1 for regression/binary classification)
    """
    
    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
    
    def forward(
        self,
        fragment_embeddings: Tensor,
        fragment_mask: Tensor
    ) -> tuple:
        """
        Compute task predictions from fragment embeddings.
        
        Args:
            fragment_embeddings: [B, max_fragments, hidden_dim]
            fragment_mask: [B, max_fragments]
            
        Returns:
            output: Molecular predictions [B, output_dim]
            contributions: Per-fragment contributions [B, max_fragments, output_dim]
        """
        contributions = self.linear(fragment_embeddings)
        contributions = contributions * fragment_mask.unsqueeze(-1)
        output = contributions.sum(dim=1)
        return output, contributions


class MultiTaskModel(nn.Module):
    """
    Multi-task model combining encoder with task-specific heads.
    
    Supports both classification (pretraining) and regression (finetuning)
    tasks with optional encoder freezing.
    
    Args:
        encoder: Fragment-aware encoder instance
        task_names: List of task names
        freeze_encoder: Whether to freeze encoder weights
        reg_encoder: L1 regularization weight for inter-fragment edges
        reg_contribution: L1 regularization weight for fragment contributions
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        task_names: List[str],
        freeze_encoder: bool = False,
        reg_encoder: float = 1e-4,
        reg_contribution: float = 0.5
    ):
        super().__init__()
        
        self.encoder = encoder
        self.task_names = list(task_names)
        self.reg_encoder = reg_encoder
        self.reg_contribution = reg_contribution
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.task_heads = nn.ModuleDict({
            name: TaskHead(encoder.hidden_dim)
            for name in task_names
        })
    
    @property
    def hidden_dim(self) -> int:
        return self.encoder.hidden_dim
    
    def forward(
        self,
        data: Data,
        task_name: str
    ) -> Dict[str, Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            data: PyG Data batch
            task_name: Name of the task to predict
            
        Returns:
            Dictionary with:
                - output: Predictions [B, 1]
                - losses: Regularization loss scalar
                - fragment_contributions: Per-fragment contributions
                - fragment_embeddings: Fragment embeddings from encoder
        """
        enc_out = self.encoder(data)
        fragment_embeddings = enc_out['fragment_embeddings']
        fragment_mask = enc_out['fragment_mask']
        reg_loss = enc_out['reg_loss']
        
        head = self.task_heads[task_name]
        output, contributions = head(fragment_embeddings, fragment_mask)
        
        num_fragments = fragment_mask.sum(dim=1, keepdim=True).clamp(min=1e-7)
        l1_contrib = (contributions.abs().sum(dim=1) / num_fragments).mean()
        
        total_reg = self.reg_encoder * reg_loss + self.reg_contribution * l1_contrib
        
        return {
            'output': output,
            'losses': total_reg,
            'fragment_contributions': contributions,
            'fragment_embeddings': fragment_embeddings,
            'x_cluster_transformed': contributions,
        }
    
    def add_task(self, task_name: str) -> None:
        """Add a new task head."""
        if task_name not in self.task_heads:
            self.task_heads[task_name] = TaskHead(self.encoder.hidden_dim)
            self.task_names.append(task_name)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder weights for finetuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class PretrainModel(MultiTaskModel):
    """
    Multi-task model for pretraining on classification tasks.
    
    Identical to MultiTaskModel but provides semantic clarity for
    the pretraining phase.
    """
    pass


class FinetuneModel(MultiTaskModel):
    """
    Multi-task model for finetuning on regression tasks.
    
    By default, freezes the encoder to only train task heads.
    
    Args:
        encoder: Pretrained encoder instance
        task_names: List of finetuning task names
        freeze_encoder: Whether to freeze encoder (default True)
        reg_encoder: L1 regularization weight for inter-fragment edges
        reg_contribution: L1 regularization weight for fragment contributions
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        task_names: List[str],
        freeze_encoder: bool = True,
        reg_encoder: float = 1e-4,
        reg_contribution: float = 0.5
    ):
        super().__init__(
            encoder=encoder,
            task_names=task_names,
            freeze_encoder=freeze_encoder,
            reg_encoder=reg_encoder,
            reg_contribution=reg_contribution
        )


def create_model(
    encoder_type: str = "gcn",
    task_names: List[str] = None,
    input_dim: int = 25,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> MultiTaskModel:
    """
    Factory function to create a complete SEAL model.
    
    Args:
        encoder_type: 'gcn' or 'gin'
        task_names: List of task names
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of encoder layers
        dropout: Dropout probability
        freeze_encoder: Whether to freeze encoder
        pretrained_path: Path to pretrained encoder weights
        **kwargs: Additional encoder arguments
        
    Returns:
        MultiTaskModel instance
    """
    if task_names is None:
        task_names = ["default"]
    
    encoder = create_encoder(
        encoder_type=encoder_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs
    )
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'encoder_state' in state_dict:
            state_dict = state_dict['encoder_state']
        elif 'model_state' in state_dict:
            encoder_state = {}
            for k, v in state_dict['model_state'].items():
                if k.startswith('encoder.'):
                    encoder_state[k[8:]] = v
            state_dict = encoder_state
        encoder.load_state_dict(state_dict)
    
    return MultiTaskModel(
        encoder=encoder,
        task_names=task_names,
        freeze_encoder=freeze_encoder,
        reg_encoder=kwargs.get('reg_encoder', 1e-4),
        reg_contribution=kwargs.get('reg_contribution', 0.5)
    )


def load_pretrained_encoder(
    checkpoint_path: str,
    encoder_type: str = "gcn",
    input_dim: int = 25,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    device: str = 'cpu',
    **kwargs
) -> BaseEncoder:
    """
    Load a pretrained encoder from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        encoder_type: 'gcn' or 'gin'
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of encoder layers
        dropout: Dropout probability
        device: Device to load weights to
        **kwargs: Additional encoder arguments
        
    Returns:
        Encoder with loaded weights
    """
    encoder = create_encoder(
        encoder_type=encoder_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(state_dict, dict):
        if 'encoder_state' in state_dict:
            state_dict = state_dict['encoder_state']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
    encoder.load_state_dict(state_dict)
    return encoder
