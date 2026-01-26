"""
SEAL-style message passing layers with fragment-aware convolution.

The key innovation is separate linear transformations for intra-fragment
(within BRICS fragments) and inter-fragment (across broken bonds) messages,
enabling interpretable fragment-level attributions.
"""

from typing import Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


class SEALConv(MessagePassing):
    """
    SEAL Convolution Layer with fragment-aware message passing.
    
    For each edge, the layer applies different linear transformations
    depending on whether the edge is within a BRICS fragment (intra)
    or crosses a fragment boundary (inter).
    
    This design enables the model to learn different representations
    for local chemical environments vs. fragment interactions, which
    is crucial for interpretability.
    
    Args:
        in_channels: Size of input features (or tuple for bipartite)
        out_channels: Size of output features
        aggr: Aggregation scheme ('mean', 'add', 'max')
        bias: Whether to include bias terms
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
        
        # Linear transform for neighbors within same fragment
        self.lin_intra = nn.Linear(in_channels[0], out_channels, bias=bias)
        
        # Linear transform for neighbors in different fragments
        self.lin_inter = nn.Linear(in_channels[0], out_channels, bias=bias)
        
        # Linear transform for self (root node)
        self.lin_root = nn.Linear(in_channels[1], out_channels, bias=False)
        
        self._edge_mask = None
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        super().reset_parameters()
        self.lin_intra.reset_parameters()
        self.lin_inter.reset_parameters()
        self.lin_root.reset_parameters()
    
    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_brics_mask: Tensor
    ) -> Tensor:
        """
        Forward pass with fragment-aware message passing.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_brics_mask: Boolean mask where True indicates edge is
                            within a fragment (not broken by BRICS)
                            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        if isinstance(x, Tensor):
            x = (x, x)
        
        # Pre-compute transformed features
        x_intra = self.lin_intra(x[0])
        x_inter = self.lin_inter(x[0])
        x_root = self.lin_root(x[1])
        
        # Store mask for message function
        self._edge_mask = edge_brics_mask
        
        # Propagate messages
        out = self.propagate(edge_index, x_intra=x_intra, x_inter=x_inter)
        
        # Add root (self) contribution
        return out + x_root
    
    def message(self, x_intra_j: Tensor, x_inter_j: Tensor) -> Tensor:
        """
        Compute messages using fragment-aware routing.
        
        Args:
            x_intra_j: Transformed features for intra-fragment edges
            x_inter_j: Transformed features for inter-fragment edges
            
        Returns:
            Messages to aggregate
        """
        # Select intra-fragment or inter-fragment representation
        # based on whether edge crosses fragment boundary
        return torch.where(
            self._edge_mask.unsqueeze(-1),
            x_intra_j,  # Edge within fragment
            x_inter_j   # Edge crosses fragment boundary
        )
    
    @property
    def inter_weights(self) -> Tensor:
        """Get inter-fragment linear weights (for regularization)."""
        return self.lin_inter.weight
    
    @property
    def inter_bias(self) -> Tensor:
        """Get inter-fragment linear bias (for regularization)."""
        return self.lin_inter.bias


class SEALGINConv(MessagePassing):
    """
    SEAL-GIN Convolution Layer with fragment-aware message passing.
    
    Uses separate MLPs for intra-fragment and inter-fragment messages,
    providing more expressive power than SEALConv while maintaining
    interpretability through fragment-level attributions.
    
    This is the GIN variant from Musial et al. (2025) extended with
    BRICS fragment awareness.
    
    Args:
        in_channels: Size of input features (or tuple for bipartite)
        out_channels: Size of output features
        aggr: Aggregation scheme (default: 'add' for GIN)
        eps: Initial epsilon value for self-loop weighting
        train_eps: Whether to learn epsilon
        bias: Whether to include bias terms
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
        
        # Learnable or fixed epsilon
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))
        
        # MLP for intra-fragment (neighbor) messages
        self.mlp_intra = nn.Sequential(
            nn.Linear(in_channels[0], out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias)
        )
        
        # MLP for inter-fragment (outside) messages
        self.mlp_inter = nn.Sequential(
            nn.Linear(in_channels[0], out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias)
        )
        
        # Linear transform for root node
        self.lin_root = nn.Linear(in_channels[1], out_channels, bias=False)
        
        self._edge_mask = None
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        super().reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        for layer in self.mlp_intra:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.mlp_inter:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.lin_root.reset_parameters()
    
    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_brics_mask: Tensor
    ) -> Tensor:
        """
        Forward pass with GIN-style fragment-aware aggregation.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_brics_mask: Boolean mask where True indicates edge is
                            within a fragment (not broken by BRICS)
                            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        if isinstance(x, Tensor):
            x = (x, x)
        
        self._edge_mask = edge_brics_mask
        
        # Aggregate neighbor messages with fragment-aware routing
        out = self.propagate(edge_index, x=x[0])
        
        # Add weighted self-loop (GIN update)
        out = out + (1 + self.eps) * self.lin_root(x[1])
        
        return out
    
    def message(self, x_j: Tensor) -> Tensor:
        """
        Compute messages using fragment-aware MLP routing.
        
        Args:
            x_j: Source node features
            
        Returns:
            Messages to aggregate
        """
        msg_intra = self.mlp_intra(x_j)
        msg_inter = self.mlp_inter(x_j)
        
        return torch.where(
            self._edge_mask.unsqueeze(-1),
            msg_intra,  # Edge within fragment
            msg_inter   # Edge crosses fragment boundary
        )
    
    @property
    def inter_weights(self) -> Tensor:
        """Get inter-fragment MLP weights (for regularization)."""
        return self.mlp_inter[0].weight
    
    @property
    def inter_bias(self) -> Tensor:
        """Get inter-fragment MLP bias (for regularization)."""
        return self.mlp_inter[0].bias


# Alias for backward compatibility
GINConv = SEALGINConv
