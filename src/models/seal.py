"""
SEAL multi-task models for pretraining and finetuning.

Implements models that combine fragment-aware encoders with
task-specific prediction heads. Fragment contributions are
directly interpretable as they sum to the final prediction.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union

from .encoder import FragmentAwareGCNEncoder, FragmentAwareGINEncoder, create_encoder


class MultiTaskPretrainModel(nn.Module):
    """
    Multi-task pretraining model for classification tasks.
    
    Uses a shared encoder and task-specific linear heads.
    Fragment contributions sum to the prediction (before sigmoid),
    enabling interpretability.
    
    Args:
        encoder: Fragment-aware encoder module
        task_names: List of task names
        regularize_encoder: L1 penalty on inter-fragment weights
        regularize_contribution: L1 penalty on fragment contributions
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        task_names: List[str],
        regularize_encoder: float = 1e-4,
        regularize_contribution: float = 0.5
    ):
        super().__init__()
        
        self.encoder = encoder
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.regularize_encoder = regularize_encoder
        self.regularize_contribution = regularize_contribution
        
        hidden_dim = encoder.hidden_features
        
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim, 1, bias=True)
            for task in task_names
        })
    
    def forward(self, data, task_name: str) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            data: PyG batch with molecular graphs
            task_name: Name of task to predict
            
        Returns:
            dict with:
                - output: Predictions [B, 1]
                - losses: Regularization loss scalar
                - fragment_contributions: Per-fragment scores [B, K, 1]
                - fragment_embeddings: Fragment representations [B, K, H]
        """
        encoder_output = self.encoder(data)
        fragment_embeddings = encoder_output['fragment_embeddings']
        fragment_mask = encoder_output['fragment_mask']
        reg_loss = encoder_output['reg_loss']
        
        # Get task head
        task_head = self.task_heads[task_name]
        
        # Compute fragment contributions
        fragment_contributions = task_head(fragment_embeddings)
        
        # Mask out non-existent fragments
        fragment_mask_expanded = fragment_mask.unsqueeze(-1)
        fragment_contributions = fragment_contributions * fragment_mask_expanded
        
        # Sum contributions to get prediction
        output = fragment_contributions.sum(dim=1)
        
        # L1 regularization on contributions
        num_fragments = fragment_mask.sum(dim=1, keepdim=True) + 1e-7
        l1_loss = (fragment_contributions.abs().sum(dim=1) / num_fragments).mean()
        
        total_reg = self.regularize_encoder * reg_loss + self.regularize_contribution * l1_loss
        
        return {
            'output': output,
            'losses': total_reg,
            'fragment_contributions': fragment_contributions,
            'fragment_embeddings': fragment_embeddings
        }


class MultiTaskRegressionModel(nn.Module):
    """
    Multi-task regression model for finetuning.
    
    Takes a pretrained encoder and adds task-specific heads.
    Supports freezing the encoder for transfer learning.
    
    Args:
        pretrained_encoder: Encoder module (pretrained or fresh)
        task_names: List of regression task names
        freeze_encoder: Whether to freeze encoder weights
        regularize_encoder: L1 penalty on inter-fragment weights
        regularize_contribution: L1 penalty on fragment contributions
    """
    
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        task_names: List[str],
        freeze_encoder: bool = False,
        regularize_encoder: float = 1e-4,
        regularize_contribution: float = 0.5
    ):
        super().__init__()
        
        self.encoder = pretrained_encoder
        self.hidden_dim = pretrained_encoder.hidden_features
        self.task_names = task_names
        self.regularize_encoder = regularize_encoder
        self.regularize_contribution = regularize_contribution
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(self.hidden_dim, 1, bias=True)
            for task in task_names
        })
    
    def forward(self, data, task_name: str) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            data: PyG batch with molecular graphs
            task_name: Name of task to predict
            
        Returns:
            dict with output, losses, x_cluster_transformed (for explanations)
        """
        encoder_output = self.encoder(data)
        fragment_embeddings = encoder_output['fragment_embeddings']
        fragment_mask = encoder_output['fragment_mask']
        reg_loss = encoder_output['reg_loss']
        
        task_head = self.task_heads[task_name]
        fragment_contributions = task_head(fragment_embeddings)
        
        fragment_mask_expanded = fragment_mask.unsqueeze(-1)
        fragment_contributions = fragment_contributions * fragment_mask_expanded
        
        output = fragment_contributions.sum(dim=1)
        
        num_fragments = fragment_mask.sum(dim=1, keepdim=True) + 1e-7
        l1_loss = (fragment_contributions.abs().sum(dim=1) / num_fragments).mean()
        
        total_reg = self.regularize_encoder * reg_loss + self.regularize_contribution * l1_loss
        
        return {
            'output': output,
            'losses': total_reg,
            'x_cluster_transformed': fragment_contributions  # For explanation extraction
        }
    
    def predict(
        self,
        data,
        task_name: str,
        return_contributions: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make predictions without computing gradients.
        
        Args:
            data: PyG batch
            task_name: Task to predict
            return_contributions: Whether to return fragment contributions
            
        Returns:
            Predictions tensor, or dict with predictions and contributions
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(data, task_name)
        
        if return_contributions:
            return {
                'predictions': out['output'],
                'fragment_contributions': out['x_cluster_transformed']
            }
        return out['output']


def build_pretrain_model(
    task_names: List[str],
    encoder_type: str = "gcn",
    input_features: int = 25,
    hidden_features: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    regularize_encoder: float = 1e-4,
    regularize_contribution: float = 0.5,
    **encoder_kwargs
) -> MultiTaskPretrainModel:
    """
    Build pretraining model with fresh encoder.
    
    Args:
        task_names: Classification task names
        encoder_type: 'gcn' or 'gin'
        input_features: Node feature dimension
        hidden_features: Hidden dimension
        num_layers: Number of encoder layers
        dropout: Dropout rate
        regularize_encoder: Encoder regularization weight
        regularize_contribution: Contribution regularization weight
        **encoder_kwargs: Additional encoder arguments
        
    Returns:
        MultiTaskPretrainModel
    """
    encoder = create_encoder(
        encoder_type=encoder_type,
        input_features=input_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout=dropout,
        **encoder_kwargs
    )
    
    return MultiTaskPretrainModel(
        encoder=encoder,
        task_names=task_names,
        regularize_encoder=regularize_encoder,
        regularize_contribution=regularize_contribution
    )


def build_finetune_model(
    task_names: List[str],
    encoder_checkpoint: Optional[str] = None,
    encoder_type: str = "gcn",
    input_features: int = 25,
    hidden_features: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    regularize_encoder: float = 1e-4,
    regularize_contribution: float = 0.5,
    device: str = "cpu",
    **encoder_kwargs
) -> MultiTaskRegressionModel:
    """
    Build finetuning model, optionally loading pretrained encoder.
    
    Args:
        task_names: Regression task names
        encoder_checkpoint: Path to pretrained encoder (None for fresh)
        encoder_type: 'gcn' or 'gin'
        input_features: Node feature dimension
        hidden_features: Hidden dimension
        num_layers: Number of encoder layers
        dropout: Dropout rate
        freeze_encoder: Whether to freeze encoder
        regularize_encoder: Encoder regularization weight
        regularize_contribution: Contribution regularization weight
        device: Device to load to
        **encoder_kwargs: Additional encoder arguments
        
    Returns:
        MultiTaskRegressionModel
    """
    encoder = create_encoder(
        encoder_type=encoder_type,
        input_features=input_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout=dropout,
        **encoder_kwargs
    )
    
    if encoder_checkpoint is not None:
        state_dict = torch.load(encoder_checkpoint, map_location=device)
        encoder.load_state_dict(state_dict)
    
    return MultiTaskRegressionModel(
        pretrained_encoder=encoder,
        task_names=task_names,
        freeze_encoder=freeze_encoder,
        regularize_encoder=regularize_encoder,
        regularize_contribution=regularize_contribution
    )
