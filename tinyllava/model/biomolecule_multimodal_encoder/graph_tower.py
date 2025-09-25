import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GINConv
import os
import pickle

class GraphTower(nn.Module):
    def __init__(self, graph_tower_cfg=None, delay_load=False, **kwargs):
        super().__init__()
        
        if graph_tower_cfg is None:
            graph_tower_cfg = {}
        
        self.graph_tower_type = graph_tower_cfg.get('graph_tower_type', 'gcn')
        self.hidden_size = graph_tower_cfg.get('hidden_size', 512)
        self.dropout = graph_tower_cfg.get('dropout', 0.3)
        self.patient_graphs_dir = graph_tower_cfg.get('patient_graphs_dir', None)
        
        self.is_loaded = False
        self.graphs_cache = {}
        
        if not delay_load:
            self.load_model()
    
    def load_model(self):
        if self.patient_graphs_dir is None:
            raise ValueError("patient_graphs_dir must be specified for graph_tower")
        
        # Load metadata to get graph info
        metadata_file = os.path.join(self.patient_graphs_dir, 'patient_graph_metadata.pkl')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Graph metadata not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Get input dimension from first graph
        if self.metadata:
            first_graph_file = os.path.join(self.patient_graphs_dir, self.metadata[0]['filename'])
            sample_graph = torch.load(first_graph_file, weights_only=False)
            input_dim = sample_graph.x.shape[1]
        else:
            input_dim = 4792  # fallback
        
        # Build GNN
        if self.graph_tower_type == 'gcn':
            self.gnn = nn.ModuleList([
                GCNConv(input_dim, self.hidden_size),
                GCNConv(self.hidden_size, self.hidden_size)
            ])
        elif self.graph_tower_type == 'gat':
            self.gnn = nn.ModuleList([
                GATConv(input_dim, self.hidden_size, heads=4, concat=False),
                GATConv(self.hidden_size, self.hidden_size, heads=1, concat=False)
            ])

        elif self.graph_tower_type == 'gin':
            # First layer
            mlp1 = nn.Sequential(
                nn.Linear(input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            # Second layer
            mlp2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.gnn = nn.ModuleList([
                GINConv(mlp1),
                GINConv(mlp2)
            ])
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.is_loaded = True
        print(f"Loaded graph tower: {self.graph_tower_type} with {len(self.metadata)} graphs")
    
    def load_patient_graph(self, sample_id):
        """Load a specific patient graph by sample ID"""
        if sample_id in self.graphs_cache:
            return self.graphs_cache[sample_id]
        
        # Find graph file for this sample
        for item in self.metadata:
            if item['sample_id'] == sample_id:
                graph_file = os.path.join(self.patient_graphs_dir, item['filename'])
                if os.path.exists(graph_file):
                    graph = torch.load(graph_file, weights_only=False)
                    self.graphs_cache[sample_id] = graph
                    return graph
        
        return None
    
    def forward(self, sample_ids):
        if not self.is_loaded:
            self.load_model()
        
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        
        batch_embeddings = []
        
        for sample_id in sample_ids:
            # Load patient's PPI graph
            graph = self.load_patient_graph(str(sample_id))
            if graph is None:
                batch_embeddings.append(torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype))
                continue
            
            # Move to device and ensure dtype
            graph = graph.to(self.device)
            x = graph.x.to(self.dtype)
            edge_index = graph.edge_index
            
            # Forward through GNN
            for i, layer in enumerate(self.gnn):
                x = layer(x, edge_index)
                if i < len(self.gnn) - 1:
                    x = torch.relu(x)
                    x = self.dropout_layer(x)
            
            # Global mean pooling to get graph-level embedding
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = global_mean_pool(x, batch)
            batch_embeddings.append(graph_embedding.squeeze(0))
        
        return torch.stack(batch_embeddings)
    
    @property
    def device(self):
        if self.is_loaded and self.gnn:
            return next(self.gnn[0].parameters()).device
        return torch.device('cpu')
    
    @property
    def dtype(self):
        if self.is_loaded and self.gnn:
            return next(self.gnn[0].parameters()).dtype
        return torch.float32