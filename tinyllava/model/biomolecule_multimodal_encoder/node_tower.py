# tinyllava/model/biomolecule_multimodal_encoder/node_tower.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os
import sys
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics.pairwise import cosine_similarity


class GCNNodeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x


class GATNodeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x


class SAGENodeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x


class NodeTower(nn.Module):
    def __init__(self, node_tower_cfg=None, delay_load=False, **kwargs):
        super().__init__()
        
        if node_tower_cfg is None:
            node_tower_cfg = {}
        
        self.is_loaded = False
        
        # Handle config attributes
        if hasattr(node_tower_cfg, 'get'):
            self.node_tower_name = node_tower_cfg.get('node_tower_type', 'gcn')
            self.num_proteins = node_tower_cfg.get('num_proteins', 4792)
            self.hidden_size = node_tower_cfg.get('hidden_size', 512)
            self.dropout = node_tower_cfg.get('dropout', 0.3)
            self.k_neighbors = node_tower_cfg.get('k_neighbors', 7)
            self.proteomics_data_path = node_tower_cfg.get('proteomics_data_path', None)
        else:
            self.node_tower_name = getattr(node_tower_cfg, 'node_tower_type', 'gcn')
            self.num_proteins = getattr(node_tower_cfg, 'num_proteins', 4792)
            self.hidden_size = getattr(node_tower_cfg, 'hidden_size', 512)
            self.dropout = getattr(node_tower_cfg, 'dropout', 0.3)
            self.k_neighbors = getattr(node_tower_cfg, 'k_neighbors', 7)
            self.proteomics_data_path = getattr(node_tower_cfg, 'proteomics_data_path', None)
        
        self.graph_data = None
        self.sample_id_to_node_idx = {}
        
        if not delay_load:
            self.load_model()
        else:
            self.node_encoder = None
    
    def load_model(self):
        # Build graph
        self._build_global_graph()
        
        # Create encoder
        if self.node_tower_name == 'gcn':
            self.node_encoder = GCNNodeEncoder(
                input_dim=self.num_proteins,
                hidden_dim=self.hidden_size,
                dropout=self.dropout
            )
        elif self.node_tower_name == 'gat':
            self.node_encoder = GATNodeEncoder(
                input_dim=self.num_proteins,
                hidden_dim=self.hidden_size,
                dropout=self.dropout
            )
        elif self.node_tower_name == 'sage':
            self.node_encoder = SAGENodeEncoder(
                input_dim=self.num_proteins,
                hidden_dim=self.hidden_size,
                dropout=self.dropout
            )
        else:
            print(f"ERROR: Unknown node tower type: {self.node_tower_name}")
            sys.exit(1)
        
        self.is_loaded = True
        print(f"Loaded node tower: {self.node_tower_name}")
    
    def _load_proteomics_data(self):
        if self.proteomics_data_path is None or self.proteomics_data_path == "":
            print("ERROR: proteomics_data_path not specified")
            sys.exit(1)
        
        # Resolve relative paths
        proteomics_path = os.path.abspath(os.path.expanduser(self.proteomics_data_path))
        
        csv_files = glob.glob(os.path.join(proteomics_path, '*.csv'))
        if not csv_files:
            print(f"ERROR: No CSV files found in {proteomics_path}")
            sys.exit(1)
        
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0)
            df.index = df.index.astype(str)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return combined_df
    
    def _build_knn_graph(self, features, k_neighbors):
        n_samples = features.shape[0]
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(features)
        np.fill_diagonal(similarity_matrix, -1)
        
        edge_list = []
        edge_weights = []
        
        for i in range(n_samples):
            similarities = similarity_matrix[i]
            neighbor_indices = np.argsort(similarities)[-k_neighbors:]
            neighbor_similarities = similarities[neighbor_indices]
            
            for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
                edge_list.append([i, neighbor_idx])
                edge_weights.append(similarity)
        
        edge_index = np.array(edge_list).T
        edge_weights = np.array(edge_weights)
        
        # Normalize to [1, 2]
        min_weight = edge_weights.min()
        max_weight = edge_weights.max()
        
        if max_weight > min_weight:
            edge_weights = (edge_weights - min_weight) / (max_weight - min_weight)
            edge_weights = edge_weights + 1.0
        else:
            edge_weights = np.full_like(edge_weights, 1.5)
        
        return edge_index, edge_weights
    
    def _build_global_graph(self):
        # Load proteomics data
        proteomics_df = self._load_proteomics_data()
        
        # Create sample ID mapping
        sample_ids = proteomics_df.index.tolist()
        self.sample_id_to_node_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        
        # Build KNN graph
        features = proteomics_df.values
        edge_index, edge_weights = self._build_knn_graph(features, self.k_neighbors)
        
        # Create PyG data object
        self.graph_data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weights, dtype=torch.float32),
            num_nodes=features.shape[0]
        )
        
        print(f"Built global graph: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
    
    def forward(self, sample_ids):
        if not self.is_loaded:
            self.load_model()
        
        # Handle single sample ID
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        elif isinstance(sample_ids, torch.Tensor):
            # Convert tensor of sample IDs to list
            sample_ids = [str(sid) for sid in sample_ids.tolist()]
        
        # Get node indices
        node_indices = []
        for sample_id in sample_ids:
            if sample_id not in self.sample_id_to_node_idx:
                print(f"ERROR: Sample ID {sample_id} not found in graph")
                sys.exit(1)
            node_indices.append(self.sample_id_to_node_idx[sample_id])
        
        # Move graph to device
        if self.graph_data.x.device != self.device:
            self.graph_data = self.graph_data.to(self.device)
        
        # Forward through GNN to get all node embeddings
        with torch.set_grad_enabled(self.training):
            all_node_embeddings = self.node_encoder(
                self.graph_data.x,
                self.graph_data.edge_index,
                self.graph_data.edge_attr
            )
        
        # Extract embeddings for requested samples
        sample_embeddings = all_node_embeddings[node_indices]
        
        return sample_embeddings
    
    @property
    def device(self):
        if self.is_loaded and self.node_encoder is not None:
            return next(self.node_encoder.parameters()).device
        return torch.device('cpu')
    
    @property
    def dtype(self):
        if self.is_loaded and self.node_encoder is not None:
            return next(self.node_encoder.parameters()).dtype
        return torch.float32
    
    @property
    def config(self):
        return {
            'hidden_size': self.hidden_size,
            'num_proteins': self.num_proteins,
            'node_tower_name': self.node_tower_name,
            'dropout': self.dropout,
            'k_neighbors': self.k_neighbors
        }


def build_node_tower(node_tower_cfg, **kwargs):
    return NodeTower(node_tower_cfg, **kwargs)