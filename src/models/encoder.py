"""
Fragment-aware encoders for SEAL framework.

Implements graph encoders that produce fragment-level embeddings
by aggregating node representations within each BRICS fragment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .layers import SEALConv, SEALGINConv


class FragmentAwareGCNEncoder(nn.Module):
    """
    GCN-based encoder with fragment-aware message passing.
    
    Uses SEALConv layers to process molecular graphs while maintaining
    separate weights for intra-fragment and inter-fragment edges.
    Outputs fragment-level embeddings by aggregating node features.
    
    Args:
        input_features: Number of input node features (default: 25)
        hidden_features: Hidden dimension size (default: 256)
        num_layers: Number of GCN layers (default: 4)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_features: int = 25,
        hidden_features: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: input -> hidden
        self.gcn_layers.append(SEALConv(input_features, hidden_features))
        self.batch_norms.append(nn.LayerNorm(hidden_features))
        
        # Remaining layers: hidden -> hidden
        for _ in range(num_layers - 1):
            self.gcn_layers.append(SEALConv(hidden_features, hidden_features))
            self.batch_norms.append(nn.LayerNorm(hidden_features))
        
        self.fragment_bn = nn.LayerNorm(hidden_features)
    
    def forward(self, data):
        """
        Encode molecular graph to fragment embeddings.
        
        Args:
            data: PyG Data object with:
                - x: Node features [N, input_features]
                - edge_index: Edge connectivity [2, E]
                - s: Fragment membership matrix [N, K]
                - mask: Edge mask (True = intra-fragment) [E]
                - batch: Batch assignment [N]
                
        Returns:
            dict with:
                - fragment_embeddings: [B, K, hidden]
                - fragment_mask: [B, K]
                - reg_loss: Regularization loss scalar
        """
        x = data.x
        edge_index = data.edge_index
        s = data.s
        batch = data.batch
        edge_brics_mask = data.mask.bool()
        
        # Message passing layers
        for conv, bn in zip(self.gcn_layers, self.batch_norms):
            x = conv(x, edge_index, edge_brics_mask)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Convert to dense batch format
        x_dense, node_mask = to_dense_batch(x, batch)
        s_dense, s_mask = to_dense_batch(s, batch)
        
        batch_size, num_nodes, _ = x_dense.size()
        
        # Apply masks
        if node_mask is not None:
            node_mask = node_mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s_mask = s_mask.view(batch_size, num_nodes, 1).to(s.dtype)
            x_dense = x_dense * node_mask
            s_dense = s_dense * s_mask
        
        # Aggregate node features to fragment level: [B, K, H]
        fragment_embeddings = torch.matmul(s_dense.transpose(1, 2), x_dense)
        fragment_embeddings = self.fragment_bn(fragment_embeddings)
        
        # Fragment mask: which fragments exist in each graph
        fragment_mask = (s_dense.sum(dim=1) > 0).float()
        
        # Compute regularization loss on inter-fragment weights
        reg_loss = 0.0
        for layer in self.gcn_layers:
            reg_loss = reg_loss + torch.norm(layer.weights_seal_outside(), p=1)
            if layer.bias_seal_outside() is not None:
                reg_loss = reg_loss + torch.norm(layer.bias_seal_outside(), p=1)
        
        return {
            'fragment_embeddings': fragment_embeddings,
            'fragment_mask': fragment_mask,
            'reg_loss': reg_loss
        }


class FragmentAwareGINEncoder(nn.Module):
    """
    GIN-based encoder with fragment-aware message passing.
    
    Uses SEALGINConv layers with MLPs instead of linear layers,
    providing increased expressiveness compared to GCN.
    
    Args:
        input_features: Number of input node features (default: 25)
        hidden_features: Hidden dimension size (default: 256)
        num_layers: Number of GIN layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        train_eps: Whether to learn epsilon parameter (default: False)
    """
    
    def __init__(
        self,
        input_features: int = 25,
        hidden_features: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        train_eps: bool = False
    ):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.train_eps = train_eps
        
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: input -> hidden
        self.gin_layers.append(
            SEALGINConv(input_features, hidden_features, aggr="add", train_eps=train_eps)
        )
        self.batch_norms.append(nn.LayerNorm(hidden_features))
        
        # Remaining layers: hidden -> hidden
        for _ in range(num_layers - 1):
            self.gin_layers.append(
                SEALGINConv(hidden_features, hidden_features, aggr="add", train_eps=train_eps)
            )
            self.batch_norms.append(nn.LayerNorm(hidden_features))
        
        self.fragment_bn = nn.LayerNorm(hidden_features)
    
    def forward(self, data):
        """
        Encode molecular graph to fragment embeddings.
        
        Args:
            data: PyG Data object (same format as GCN encoder)
                
        Returns:
            dict with fragment_embeddings, fragment_mask, reg_loss
        """
        x = data.x
        edge_index = data.edge_index
        s = data.s
        batch = data.batch
        edge_brics_mask = data.mask.bool()
        
        # Message passing layers
        for conv, bn in zip(self.gin_layers, self.batch_norms):
            x = conv(x, edge_index, edge_brics_mask)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Convert to dense batch format
        x_dense, node_mask = to_dense_batch(x, batch)
        s_dense, s_mask = to_dense_batch(s, batch)
        
        batch_size, num_nodes, _ = x_dense.size()
        
        if node_mask is not None:
            node_mask = node_mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s_mask = s_mask.view(batch_size, num_nodes, 1).to(s.dtype)
            x_dense = x_dense * node_mask
            s_dense = s_dense * s_mask
        
        # Aggregate to fragment level
        fragment_embeddings = torch.matmul(s_dense.transpose(1, 2), x_dense)
        fragment_embeddings = self.fragment_bn(fragment_embeddings)
        
        fragment_mask = (s_dense.sum(dim=1) > 0).float()
        
        # Regularization loss
        reg_loss = 0.0
        for layer in self.gin_layers:
            reg_loss = reg_loss + torch.norm(layer.weights_seal_outside(), p=1)
            if layer.bias_seal_outside() is not None:
                reg_loss = reg_loss + torch.norm(layer.bias_seal_outside(), p=1)
        
        return {
            'fragment_embeddings': fragment_embeddings,
            'fragment_mask': fragment_mask,
            'reg_loss': reg_loss
        }


def create_encoder(
    encoder_type: str = "gcn",
    input_features: int = 25,
    hidden_features: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    **kwargs
):
    """
    Factory function to create encoder by type.
    
    Args:
        encoder_type: 'gcn' or 'gin'
        input_features: Input feature dimension
        hidden_features: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        **kwargs: Additional arguments (e.g., train_eps for GIN)
        
    Returns:
        Encoder module
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "gcn":
        return FragmentAwareGCNEncoder(
            input_features=input_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout
        )
    elif encoder_type == "gin":
        return FragmentAwareGINEncoder(
            input_features=input_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            train_eps=kwargs.get('train_eps', False)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'gcn' or 'gin'.")


def load_encoder(
    checkpoint_path: str,
    encoder_type: str = "gcn",
    input_features: int = 25,
    hidden_features: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    device: str = "cpu",
    **kwargs
):
    """
    Load pretrained encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt file with encoder state dict
        encoder_type: 'gcn' or 'gin'
        input_features: Must match saved model
        hidden_features: Must match saved model
        num_layers: Must match saved model
        dropout: Dropout rate
        device: Device to load to
        **kwargs: Additional encoder arguments
        
    Returns:
        Loaded encoder module
    """
    encoder = create_encoder(
        encoder_type=encoder_type,
        input_features=input_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    
    return encoder
