"""
SEAL models for multitask molecular property prediction.

Provides models for both pretraining (classification) and fine-tuning
(regression) with shared fragment-aware encoders and task-specific heads.
"""

from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import FragmentAwareEncoder


class TaskHead(nn.Module):
    """
    Simple linear head for fragment-level prediction.
    
    Maps fragment embeddings to scalar contributions that sum
    to the final prediction, enabling fragment-level attributions.
    
    Args:
        hidden_dim: Dimension of fragment embeddings
        output_dim: Output dimension (1 for regression/binary classification)
    """
    
    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, fragment_embeddings: Tensor) -> Tensor:
        """
        Compute per-fragment contributions.
        
        Args:
            fragment_embeddings: [batch, num_fragments, hidden_dim]
            
        Returns:
            Fragment contributions: [batch, num_fragments, output_dim]
        """
        return self.linear(fragment_embeddings)


class MLPHead(nn.Module):
    """
    MLP head with hidden layer for more complex predictions.
    
    Args:
        hidden_dim: Dimension of fragment embeddings
        output_dim: Output dimension
        mlp_hidden: Hidden layer dimension (default: hidden_dim // 2)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        mlp_hidden: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if mlp_hidden is None:
            mlp_hidden = hidden_dim // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_dim)
        )
    
    def forward(self, fragment_embeddings: Tensor) -> Tensor:
        return self.mlp(fragment_embeddings)


class MultiTaskModel(nn.Module):
    """
    Base class for multitask SEAL models.
    
    Combines a shared fragment-aware encoder with task-specific
    prediction heads. Supports both pretraining and fine-tuning.
    
    Args:
        encoder: FragmentAwareEncoder instance
        task_names: List of task names
        head_type: Type of task head ('linear' or 'mlp')
        regularize_encoder: L1 regularization weight for encoder
        regularize_contribution: L1 regularization for fragment contributions
    """
    
    def __init__(
        self,
        encoder: FragmentAwareEncoder,
        task_names: List[str],
        head_type: str = 'linear',
        regularize_encoder: float = 1e-4,
        regularize_contribution: float = 0.5
    ):
        super().__init__()
        
        self.encoder = encoder
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.hidden_dim = encoder.hidden_features
        self.regularize_encoder = regularize_encoder
        self.regularize_contribution = regularize_contribution
        
        # Create task-specific heads
        HeadClass = TaskHead if head_type == 'linear' else MLPHead
        self.task_heads = nn.ModuleDict({
            name: HeadClass(self.hidden_dim) for name in task_names
        })
    
    def forward(self, data, task_name: str) -> Dict[str, Any]:
        """
        Forward pass for a specific task.
        
        Args:
            data: PyG Data/Batch object
            task_name: Name of the task to predict
            
        Returns:
            Dictionary containing:
                - output: Predictions [batch, 1]
                - losses: Regularization losses
                - fragment_contributions: Per-fragment contributions
                - fragment_embeddings: Fragment representations
        """
        # Encode molecular graph
        encoder_output = self.encoder(data)
        fragment_embeddings = encoder_output['fragment_embeddings']
        fragment_mask = encoder_output['fragment_mask']
        reg_loss = encoder_output['reg_loss']
        
        # Get task-specific head
        task_head = self.task_heads[task_name]
        
        # Compute per-fragment contributions
        fragment_contributions = task_head(fragment_embeddings)
        
        # Mask invalid fragments
        fragment_mask_expanded = fragment_mask.unsqueeze(-1)
        fragment_contributions = fragment_contributions * fragment_mask_expanded
        
        # Sum contributions for final prediction
        output = fragment_contributions.sum(dim=1)
        
        # Compute contribution regularization (sparsity)
        num_fragments = fragment_mask.sum(dim=1, keepdim=True) + 1e-7
        l1_contrib = (fragment_contributions.abs().sum(dim=1) / num_fragments).mean()
        
        # Total regularization loss
        total_reg = (
            self.regularize_encoder * reg_loss +
            self.regularize_contribution * l1_contrib
        )
        
        return {
            'output': output,
            'losses': total_reg,
            'fragment_contributions': fragment_contributions,
            'fragment_embeddings': fragment_embeddings,
            'fragment_mask': fragment_mask
        }
    
    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class PretrainModel(MultiTaskModel):
    """
    Model for pretraining on classification tasks.
    
    Uses BCE loss with logits for binary classification tasks.
    Designed for pretraining on multiple ADME classification endpoints.
    """
    
    def __init__(
        self,
        encoder: FragmentAwareEncoder,
        task_names: List[str],
        regularize_encoder: float = 1e-4,
        regularize_contribution: float = 0.5
    ):
        super().__init__(
            encoder=encoder,
            task_names=task_names,
            head_type='linear',
            regularize_encoder=regularize_encoder,
            regularize_contribution=regularize_contribution
        )


class RegressionModel(MultiTaskModel):
    """
    Model for fine-tuning on regression tasks.
    
    Uses MSE loss for continuous property prediction.
    Designed for fine-tuning on ADME regression endpoints.
    
    Args:
        encoder: Pretrained FragmentAwareEncoder
        task_names: List of regression task names
        freeze_encoder: Whether to freeze encoder weights
        regularize_encoder: L1 regularization for encoder
        regularize_contribution: L1 regularization for contributions
    """
    
    def __init__(
        self,
        encoder: FragmentAwareEncoder,
        task_names: List[str],
        freeze_encoder: bool = False,
        regularize_encoder: float = 1e-4,
        regularize_contribution: float = 0.5
    ):
        super().__init__(
            encoder=encoder,
            task_names=task_names,
            head_type='linear',
            regularize_encoder=regularize_encoder,
            regularize_contribution=regularize_contribution
        )
        
        if freeze_encoder:
            self.freeze_encoder()


def build_model(
    task_names: List[str],
    input_features: int = 25,
    hidden_features: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    conv_type: str = 'seal',
    model_type: str = 'regression',
    pretrained_encoder_path: Optional[str] = None,
    freeze_encoder: bool = False,
    device: str = 'cpu'
) -> MultiTaskModel:
    """
    Factory function to build SEAL model.
    
    Args:
        task_names: List of task names
        input_features: Node feature dimension
        hidden_features: Hidden dimension
        num_layers: Number of conv layers
        dropout: Dropout rate
        conv_type: 'seal' or 'gin'
        model_type: 'pretrain' or 'regression'
        pretrained_encoder_path: Path to pretrained encoder
        freeze_encoder: Whether to freeze encoder
        device: Device to create model on
        
    Returns:
        Configured MultiTaskModel
    """
    # Create or load encoder
    encoder = FragmentAwareEncoder(
        input_features=input_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type
    )
    
    if pretrained_encoder_path:
        state_dict = torch.load(pretrained_encoder_path, map_location=device)
        encoder.load_state_dict(state_dict)
    
    # Create model
    if model_type == 'pretrain':
        model = PretrainModel(encoder=encoder, task_names=task_names)
    else:
        model = RegressionModel(
            encoder=encoder,
            task_names=task_names,
            freeze_encoder=freeze_encoder
        )
    
    return model.to(device)
