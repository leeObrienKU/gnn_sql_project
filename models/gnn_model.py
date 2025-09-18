import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# used for reference code https://github.com/sailab-code/gnn/blob/master/GNN.py


class GNN(torch.nn.Module):
    def __init__(self, model_type='GCN', input_dim=4, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.5, num_heads=4):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout_rate = dropout  # Store dropout rate
        self.num_heads = num_heads  # Store number of attention heads
        
        # Create list to hold all layers
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=self.num_heads, dropout=dropout))
            hidden_dim = hidden_dim * self.num_heads  # Account for concatenated heads
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
        # Hidden layers
        for _ in range(num_layers - 2):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=self.num_heads, dropout=dropout))
                hidden_dim = hidden_dim * self.num_heads
            elif model_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
        # Output layer
        if model_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            
        # Remove hardcoded dropout - passing the value from the constructor now
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Process through all layers except last
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        if self.model_type == 'GAT':
            x = F.elu(x)
            
        return F.log_softmax(x, dim=1)