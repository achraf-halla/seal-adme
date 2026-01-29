"""
Graph neural network layers for SEAL framework.

Implements fragment-aware message passing with separate transformations
for intra-fragment and inter-fragment edges.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SEALConv(MessagePassing):
    """
    SEAL-style GCN convolution with separate intra/inter-fragment weights.
    
    For edges within fragments, uses lin_neighbours.
    For edges crossing fragment boundaries, uses lin_outside.
    This enables interpretable fragment-level attributions.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation scheme ('mean', 'add', 'max')
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.lin_neighbours = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_outside = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels[1], out_channels, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_neighbours.reset_parameters()
        self.lin_outside.reset_parameters()
        self.lin_root.reset_parameters()
    
    def forward(self, x, edge_index, edge_brics_mask):
        """
        Forward pass with fragment-aware message passing.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_brics_mask: Boolean mask, True for intra-fragment edges
            
        Returns:
            Updated node features [N, out_channels]
        """
        if isinstance(x, torch.Tensor):
            x = (x, x)
        
        x_in = self.lin_neighbours(x[0])
        x_out = self.lin_outside(x[0])
        x_root = self.lin_root(x[1])
        
        self._edge_brics_mask = edge_brics_mask
        out = self.propagate(edge_index, x_in=x_in, x_out=x_out)
        
        return out + x_root
    
    def message(self, x_in_j, x_out_j):
        """Select message based on edge type."""
        return torch.where(
            self._edge_brics_mask.unsqueeze(-1),
            x_in_j,
            x_out_j
        )
    
    def weights_seal_outside(self):
        """Get inter-fragment weights for regularization."""
        return self.lin_outside.weight
    
    def bias_seal_outside(self):
        """Get inter-fragment bias for regularization."""
        return self.lin_outside.bias


class SEALGINConv(MessagePassing):
    """
    SEAL-style GIN convolution with separate intra/inter-fragment MLPs.
    
    Uses Graph Isomorphism Network architecture with learnable epsilon,
    but applies different MLPs for intra-fragment vs inter-fragment edges.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation scheme (default 'add' for GIN)
        eps: Initial epsilon value
        train_eps: Whether epsilon is trainable
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "add",
        eps: float = 1e-3,
        train_eps: bool = False,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        # MLP for intra-fragment messages
        self.mlp_neighbours = nn.Sequential(
            nn.Linear(in_channels[0], out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias)
        )
        
        # MLP for inter-fragment messages
        self.mlp_outside = nn.Sequential(
            nn.Linear(in_channels[0], out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias)
        )
        
        self.lin_root = nn.Linear(in_channels[1], out_channels, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        for layer in self.mlp_neighbours:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.mlp_outside:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.lin_root.reset_parameters()
    
    def forward(self, x, edge_index, edge_brics_mask):
        """
        Forward pass with fragment-aware GIN message passing.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_brics_mask: Boolean mask, True for intra-fragment edges
            
        Returns:
            Updated node features [N, out_channels]
        """
        if isinstance(x, torch.Tensor):
            x = (x, x)
        
        self._edge_brics_mask = edge_brics_mask
        out = self.propagate(edge_index, x=x[0])
        out = out + (1 + self.eps) * self.lin_root(x[1])
        
        return out
    
    def message(self, x_j):
        """Apply appropriate MLP based on edge type."""
        msg_neighbours = self.mlp_neighbours(x_j)
        msg_outside = self.mlp_outside(x_j)
        return torch.where(
            self._edge_brics_mask.unsqueeze(-1),
            msg_neighbours,
            msg_outside
        )
    
    def weights_seal_outside(self):
        """Get first layer weights of inter-fragment MLP for regularization."""
        return self.mlp_outside[0].weight
    
    def bias_seal_outside(self):
        """Get first layer bias of inter-fragment MLP for regularization."""
        return self.mlp_outside[0].bias
