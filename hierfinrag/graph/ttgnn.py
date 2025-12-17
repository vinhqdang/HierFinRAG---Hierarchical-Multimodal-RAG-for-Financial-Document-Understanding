import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class TTGNN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Node type embeddings
        # P:0, S:1, T:2, C:3, G:4
        self.node_type_emb = nn.Embedding(5, hidden_dim)  
        
        # Edge type embeddings
        # sem:0, struct:1, ref:2, temp:3, acc:4
        self.edge_type_emb = nn.Embedding(5, hidden_dim)  
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=8,
                edge_dim=hidden_dim,
                concat=False
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Input projection if needed
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
    def forward(self, x, edge_index, edge_attr, node_types):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge types [num_edges]
            node_types: Node type IDs [num_nodes]
        
        Returns:
            Updated node embeddings [num_nodes, hidden_dim]
        """
        # Add node type information
        h = self.input_proj(x) + self.node_type_emb(node_types)
        
        # Message passing
        for gat in self.gat_layers:
            edge_embeddings = self.edge_type_emb(edge_attr)
            h = gat(h, edge_index, edge_attr=edge_embeddings)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        # Output projection
        h = self.output_proj(h)
        
        return h
