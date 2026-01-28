"""
Fragment-aware molecular encoders for SEAL.

This module provides encoder architectures that transform molecular graphs
into fragment-level embeddings using BRICS decomposition information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from typing import Dict, Optional, Union

from .layers import SEALConv, SEALGINConv


class EncoderOutput:
    """Container for encoder outputs."""
    
    def __init__(
        self,
        fragment_embeddings: Tensor,
        fragment_mask: Tensor,
        reg_loss: Tensor,
        node_embeddings: Optional[Tensor] = None
    ):
        self.fragment_embeddings = fragment_embeddings
        self.fragment_mask = fragment_mask
        self.reg_loss = reg_loss
        self.node_embeddings = node_embeddings
    
    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)


class BaseEncoder(nn.Module):
    """
    Base class for fragment-aware molecular encoders.
    
    Subclasses must implement _build_conv_layer() to define the
    specific convolution type (GCN, GIN, etc.).
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of message passing layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 25,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.conv_layers.append(self._build_conv_layer(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(self._build_conv_layer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.fragment_norm = nn.LayerNorm(hidden_dim)
    
    def _build_conv_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_conv_layer")
    
    def _compute_reg_loss(self) -> Tensor:
        """Compute L1 regularization on inter-fragment weights."""
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.conv_layers:
            reg_loss = reg_loss + torch.norm(layer.inter_weight, p=1)
            if layer.inter_bias is not None:
                reg_loss = reg_loss + torch.norm(layer.inter_bias, p=1)
        return reg_loss
    
    def _aggregate_to_fragments(
        self,
        x: Tensor,
        s: Tensor,
        batch: Tensor
    ) -> tuple:
        """
        Aggregate node embeddings to fragment-level using assignment matrix.
        
        Args:
            x: Node embeddings [N, hidden_dim]
            s: Fragment assignment matrix [N, max_fragments]
            batch: Batch assignment vector [N]
            
        Returns:
            fragment_embeddings: [B, max_fragments, hidden_dim]
            fragment_mask: [B, max_fragments]
        """
        x_dense, node_mask = to_dense_batch(x, batch)
        s_dense, _ = to_dense_batch(s, batch)
        
        batch_size, num_nodes, _ = x_dense.size()
        
        if node_mask is not None:
            node_mask_expanded = node_mask.view(batch_size, num_nodes, 1).float()
            x_dense = x_dense * node_mask_expanded
            s_dense = s_dense * node_mask_expanded
        
        fragment_embeddings = torch.matmul(s_dense.transpose(1, 2), x_dense)
        fragment_embeddings = self.fragment_norm(fragment_embeddings)
        
        fragment_mask = (s_dense.sum(dim=1) > 0).float()
        
        return fragment_embeddings, fragment_mask
    
    def forward(self, data: Data) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with attributes:
                - x: Node features [N, input_dim]
                - edge_index: Edge connectivity [2, E]
                - s: Fragment assignment matrix [N, max_fragments]
                - batch: Batch assignment [N]
                - mask: Edge mask for intra-fragment edges [E]
                
        Returns:
            Dictionary with:
                - fragment_embeddings: [B, max_fragments, hidden_dim]
                - fragment_mask: [B, max_fragments]
                - reg_loss: Scalar regularization loss
        """
        x = data.x
        edge_index = data.edge_index
        edge_mask = data.mask.bool()
        
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x, edge_index, edge_mask)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        fragment_embeddings, fragment_mask = self._aggregate_to_fragments(
            x, data.s, data.batch
        )
        
        reg_loss = self._compute_reg_loss()
        
        return {
            'fragment_embeddings': fragment_embeddings,
            'fragment_mask': fragment_mask,
            'reg_loss': reg_loss,
            'node_embeddings': x
        }


class GCNEncoder(BaseEncoder):
    """
    Fragment-aware GCN encoder.
    
    Uses SEALConv layers with mean aggregation and linear transformations.
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of message passing layers
        dropout: Dropout probability
    """
    
    def _build_conv_layer(self, in_channels: int, out_channels: int) -> SEALConv:
        return SEALConv(in_channels, out_channels, aggr="mean")


class GINEncoder(BaseEncoder):
    """
    Fragment-aware GIN encoder.
    
    Uses SEALGINConv layers with sum aggregation and MLPs.
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of message passing layers
        dropout: Dropout probability
        train_eps: Whether to make epsilon learnable
    """
    
    def __init__(
        self,
        input_dim: int = 25,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        train_eps: bool = False
    ):
        self.train_eps = train_eps
        super().__init__(input_dim, hidden_dim, num_layers, dropout)
    
    def _build_conv_layer(self, in_channels: int, out_channels: int) -> SEALGINConv:
        return SEALGINConv(
            in_channels,
            out_channels,
            aggr="add",
            train_eps=self.train_eps
        )


# Aliases for backward compatibility
FragmentAwareEncoder = GCNEncoder
SEALGINEncoder = GINEncoder


def create_encoder(
    encoder_type: str = "gcn",
    input_dim: int = 25,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    **kwargs
) -> BaseEncoder:
    """
    Factory function to create encoder by type.
    
    Args:
        encoder_type: 'gcn' or 'gin'
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of message passing layers
        dropout: Dropout probability
        **kwargs: Additional arguments for specific encoders
        
    Returns:
        Encoder instance
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "gcn":
        return GCNEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif encoder_type == "gin":
        train_eps = kwargs.get("train_eps", False)
        return GINEncoder(input_dim, hidden_dim, num_layers, dropout, train_eps)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'gcn' or 'gin'.")
