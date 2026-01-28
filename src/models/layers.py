"""
Custom message-passing layers for fragment-aware molecular encoding.

This module provides SEAL (Substructure-aware Explainable ADME Learning)
convolution layers that differentiate between intra-fragment and
inter-fragment edges during message passing.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple, Union


class SEALConv(MessagePassing):
    """
    Fragment-aware graph convolution layer.
    
    Applies separate linear transformations to messages from neighbors
    within the same BRICS fragment versus neighbors in different fragments.
    Uses mean aggregation by default.
    
    Args:
        in_channels: Size of input node features
        out_channels: Size of output node features
        aggr: Aggregation scheme ('mean', 'add', 'max')
        bias: If True, add learnable bias
    """
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "mean",
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin_intra = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_inter = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels[1], out_channels, bias=False)
        
        self._edge_mask: Optional[Tensor] = None
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_intra.reset_parameters()
        self.lin_inter.reset_parameters()
        self.lin_root.reset_parameters()
    
    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_mask: Tensor
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels] or tuple for bipartite graphs
            edge_index: Edge connectivity [2, E]
            edge_mask: Boolean mask, True for intra-fragment edges [E]
            
        Returns:
            Updated node features [N, out_channels]
        """
        if isinstance(x, Tensor):
            x = (x, x)
        
        x_intra = self.lin_intra(x[0])
        x_inter = self.lin_inter(x[0])
        x_root = self.lin_root(x[1])
        
        self._edge_mask = edge_mask
        out = self.propagate(edge_index, x_intra=x_intra, x_inter=x_inter)
        
        return out + x_root
    
    def message(self, x_intra_j: Tensor, x_inter_j: Tensor) -> Tensor:
        mask = self._edge_mask.unsqueeze(-1)
        return torch.where(mask, x_intra_j, x_inter_j)
    
    @property
    def inter_weight(self) -> Tensor:
        return self.lin_inter.weight
    
    @property
    def inter_bias(self) -> Optional[Tensor]:
        return self.lin_inter.bias


class SEALGINConv(MessagePassing):
    """
    Fragment-aware Graph Isomorphism Network convolution layer.
    
    Extends GIN with separate MLPs for intra-fragment and inter-fragment
    messages. Uses sum aggregation and a learnable epsilon for self-loops.
    
    Args:
        in_channels: Size of input node features
        out_channels: Size of output node features
        aggr: Aggregation scheme (default 'add' for GIN)
        eps: Initial epsilon value for self-loop weighting
        train_eps: If True, epsilon is learnable
        bias: If True, add learnable bias to linear layers
    """
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
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
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_eps = eps
        
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))
        
        self.mlp_intra = nn.Sequential(
            nn.Linear(in_channels[0], out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias)
        )
        
        self.mlp_inter = nn.Sequential(
            nn.Linear(in_channels[0], out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias)
        )
        
        self.lin_root = nn.Linear(in_channels[1], out_channels, bias=False)
        
        self._edge_mask: Optional[Tensor] = None
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        if isinstance(self.eps, nn.Parameter):
            self.eps.data.fill_(self.initial_eps)
        else:
            self.eps.fill_(self.initial_eps)
        
        for module in self.mlp_intra:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.mlp_inter:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.lin_root.reset_parameters()
    
    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_mask: Tensor
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels] or tuple for bipartite graphs
            edge_index: Edge connectivity [2, E]
            edge_mask: Boolean mask, True for intra-fragment edges [E]
            
        Returns:
            Updated node features [N, out_channels]
        """
        if isinstance(x, Tensor):
            x = (x, x)
        
        self._edge_mask = edge_mask
        out = self.propagate(edge_index, x=x[0])
        out = out + (1 + self.eps) * self.lin_root(x[1])
        
        return out
    
    def message(self, x_j: Tensor) -> Tensor:
        msg_intra = self.mlp_intra(x_j)
        msg_inter = self.mlp_inter(x_j)
        mask = self._edge_mask.unsqueeze(-1)
        return torch.where(mask, msg_intra, msg_inter)
    
    @property
    def inter_weight(self) -> Tensor:
        return self.mlp_inter[0].weight
    
    @property
    def inter_bias(self) -> Optional[Tensor]:
        return self.mlp_inter[0].bias
