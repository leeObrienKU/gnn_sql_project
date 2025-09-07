import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HeteroConv, Linear

class GNN(torch.nn.Module):
    """Base GNN model supporting GCN, GAT, and GraphSAGE architectures"""
    def __init__(
        self,
        model_type='GCN',
        input_dim=4,
        hidden_dim=64,
        output_dim=2,
        num_layers=2,
        dropout=0.5,
        heads=4  # For GAT
    ):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create list of convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First layer (input -> hidden)
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers (hidden -> hidden)
        for _ in range(num_layers - 2):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
            elif model_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer (hidden -> output)
        if model_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(hidden_dim, output_dim))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Process through layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class HeteroGNN(torch.nn.Module):
    """Heterogeneous GNN for multi-type nodes and edges"""
    def __init__(
        self,
        hidden_dim=64,
        output_dim=2,
        num_layers=2,
        dropout=0.5,
        heads=4  # For GAT
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create convolution layers
        self.convs = torch.nn.ModuleList()
        
        # Define edge types for message passing
        metadata = (
            ['employee', 'department', 'title'],  # Node types
            [
                ('employee', 'works_in', 'department'),
                ('department', 'rev_works_in', 'employee'),
                ('employee', 'has_role', 'title'),
                ('title', 'rev_has_role', 'employee')
            ]  # Edge types
        )
        
        # Create heterogeneous convolution layers
        for _ in range(num_layers):
            conv_dict = {
                ('employee', 'works_in', 'department'): 
                    GATConv((-1, -1), hidden_dim // heads, heads=heads),
                ('department', 'rev_works_in', 'employee'): 
                    GATConv((-1, -1), hidden_dim // heads, heads=heads),
                ('employee', 'has_role', 'title'): 
                    GATConv((-1, -1), hidden_dim // heads, heads=heads),
                ('title', 'rev_has_role', 'employee'): 
                    GATConv((-1, -1), hidden_dim // heads, heads=heads)
            }
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)
        
        # Final prediction layer for employee nodes
        self.lin = Linear(hidden_dim, output_dim)
    
    def forward(self, x_dict, edge_index_dict):
        # Process through heterogeneous convolution layers
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Final layer
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        
        # Predict on employee nodes
        out = self.lin(x_dict['employee'])
        return F.log_softmax(out, dim=1)
