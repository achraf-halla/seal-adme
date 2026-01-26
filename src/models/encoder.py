"""
Fragment-aware molecular encoders for SEAL framework.

These encoders process molecular graphs and produce fragment-level
embeddings that can be used for interpretable property prediction.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_batch

from .layers import SEALConv, GINConv


class FragmentAwareEncoder(nn.Module):
    """
    Encoder that produces fragment-level molecular representations.
    
    Uses SEAL-style message passing with separate transformations for
    intra-fragment and inter-fragment edges, followed by fragment
    pooling using the BRICS decomposition matrix S.
    
    Args:
        input_features: Dimension of input node features
        hidden_features: Dimension of hidden representations
        num_layers: Number of message passing layers
        dropout: Dropout probability
        conv_type: Type of convolution ('seal' or 'gin')
    """
    
    def __init__(
        self,
        input_features: int = 25,
        hidden_features: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        conv_type: str = 'seal'
    ):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.conv_type = conv_type
        
        self.dropout = nn.Dropout(dropout)
        
        # Build convolution layers
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Choose convolution type
        ConvClass = SEALConv if conv_type == 'seal' else GINConv
        
        # First layer: input_features -> hidden_features
        self.conv_layers.append(ConvClass(input_features, hidden_features))
        self.norms.append(nn.LayerNorm(hidden_features))
        
        # Remaining layers: hidden_features -> hidden_features
        for _ in range(num_layers - 1):
            self.conv_layers.append(ConvClass(hidden_features, hidden_features))
            self.norms.append(nn.LayerNorm(hidden_features))
        
        # Normalization for fragment embeddings
        self.fragment_norm = nn.LayerNorm(hidden_features)
    
    def forward(self, data) -> Dict[str, Any]:
        """
        Encode molecular graph to fragment embeddings.
        
        Args:
            data: PyG Data/Batch object with:
                - x: Node features [num_nodes, input_features]
                - edge_index: Edge connectivity [2, num_edges]
                - s: Fragment membership matrix [num_nodes, num_fragments]
                - mask: Edge mask (1 = intra-fragment, 0 = inter-fragment)
                - batch: Batch assignment for nodes
                
        Returns:
            Dictionary containing:
                - fragment_embeddings: [batch_size, max_fragments, hidden_features]
                - fragment_mask: [batch_size, max_fragments]
                - node_embeddings: [num_nodes, hidden_features]
                - reg_loss: L1 regularization loss for inter-fragment weights
        """
        x = data.x
        edge_index = data.edge_index
        s = data.s
        batch = data.batch
        
        # Convert edge mask to boolean (True = within fragment)
        edge_brics_mask = data.mask.bool()
        
        # Message passing layers
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x, edge_index, edge_brics_mask)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Store node embeddings before pooling
        node_embeddings = x
        
        # Convert to dense batch format
        x_dense, node_mask = to_dense_batch(x, batch)
        s_dense, _ = to_dense_batch(s, batch)
        
        batch_size, num_nodes, _ = x_dense.size()
        
        # Apply node mask
        if node_mask is not None:
            node_mask_expanded = node_mask.unsqueeze(-1).float()
            x_dense = x_dense * node_mask_expanded
        
        # Fragment pooling: S^T @ X
        # s_dense: [batch, nodes, fragments]
        # x_dense: [batch, nodes, hidden]
        # result: [batch, fragments, hidden]
        fragment_embeddings = torch.matmul(
            s_dense.transpose(1, 2),  # [batch, fragments, nodes]
            x_dense                    # [batch, nodes, hidden]
        )
        
        # Normalize fragment embeddings
        fragment_embeddings = self.fragment_norm(fragment_embeddings)
        
        # Create fragment mask (1 if fragment has any atoms)
        fragment_mask = (s_dense.sum(dim=1) > 0).float()
        
        # Compute regularization loss on inter-fragment weights
        reg_loss = self._compute_regularization()
        
        return {
            'fragment_embeddings': fragment_embeddings,
            'fragment_mask': fragment_mask,
            'node_embeddings': node_embeddings,
            'reg_loss': reg_loss
        }
    
    def _compute_regularization(self) -> Tensor:
        """Compute L1 regularization on inter-fragment weights."""
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for layer in self.conv_layers:
            if hasattr(layer, 'inter_weights'):
                reg_loss = reg_loss + torch.norm(layer.inter_weights, p=1)
            if hasattr(layer, 'inter_bias') and layer.inter_bias is not None:
                reg_loss = reg_loss + torch.norm(layer.inter_bias, p=1)
        
        return reg_loss


class GINEncoder(FragmentAwareEncoder):
    """
    GIN-based fragment-aware encoder.
    
    Uses Graph Isomorphism Network convolutions with separate MLPs
    for intra-fragment and inter-fragment messages. More expressive
    than SEAL-GCN but maintains interpretability.
    """
    
    def __init__(
        self,
        input_features: int = 25,
        hidden_features: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__(
            input_features=input_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            conv_type='gin'
        )


def load_encoder(
    checkpoint_path: str,
    input_features: int = 25,
    hidden_features: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    conv_type: str = 'seal',
    device: str = 'cpu'
) -> FragmentAwareEncoder:
    """
    Load a pretrained encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to saved encoder state dict
        input_features: Input feature dimension
        hidden_features: Hidden feature dimension
        num_layers: Number of conv layers
        dropout: Dropout rate
        conv_type: Type of convolution ('seal' or 'gin')
        device: Device to load model on
        
    Returns:
        Loaded encoder model
    """
    encoder = FragmentAwareEncoder(
        input_features=input_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)
    
    return encoder
