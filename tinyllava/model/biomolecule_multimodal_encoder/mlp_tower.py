import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

class MLP3(nn.Module):
    """3-layer MLP for protein expression encoding (from your baseline_models.py)"""
    
    def __init__(self, input_dim, output_dim=256, dropout=0.3):
        super(MLP3, self).__init__()
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, output_dim)  
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.layers(x)


class MLP5(nn.Module):
    """5-layer MLP for protein expression encoding (from your baseline_models.py)"""
    
    def __init__(self, input_dim, output_dim=256, dropout=0.3):
        super(MLP5, self).__init__()
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, output_dim)  
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.layers(x)


class MLPTower(nn.Module):
    """MLP Tower for protein expression encoding - replaces vision tower in TinyLLaVA"""
    
    def __init__(self, mlp_tower_cfg=None, delay_load=False, **kwargs):
        super().__init__()
        
        if mlp_tower_cfg is None:
            mlp_tower_cfg = {}
        
        self.is_loaded = False
        self.mlp_tower_name = mlp_tower_cfg.get('mm_mlp_tower', 'mlp_3')
        self.select_layer = mlp_tower_cfg.get('mm_mlp_select_layer', -1) 
        self.select_feature = mlp_tower_cfg.get('mm_mlp_select_feature', 'cls')
        
        self.num_proteins = mlp_tower_cfg.get('num_proteins', 4792)  
        self.hidden_size = mlp_tower_cfg.get('hidden_size', 256)
        self.dropout = mlp_tower_cfg.get('dropout', 0.3)
        
        if not delay_load:
            self.load_model()
        else:
            self.mlp_encoder = None
    
    def load_model(self):
        
        if self.mlp_tower_name == 'mlp_3':
            self.mlp_encoder = MLP3(
                input_dim=self.num_proteins,
                output_dim=self.hidden_size,
                dropout=self.dropout
            )
        elif self.mlp_tower_name == 'mlp_5':
            self.mlp_encoder = MLP5(
                input_dim=self.num_proteins, 
                output_dim=self.hidden_size,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown MLP tower type: {self.mlp_tower_name}")
        
        self.is_loaded = True
        print(f"Loaded MLP tower: {self.mlp_tower_name}")
    
    def forward(self, protein_expressions):
        """
        Forward pass through MLP tower
        
        Args:
            protein_expressions: Tensor of shape (batch_size, num_proteins) or list of tensors
        
        Returns:
            protein_features: Tensor of shape (batch_size, hidden_size)
        """
        if not self.is_loaded:
            self.load_model()
        
        # Handle different input formats
        if isinstance(protein_expressions, list):
            # Convert list of tensors to batched tensor
            if len(protein_expressions) == 0:
                return torch.zeros(0, self.hidden_size, device=self.device, dtype=self.dtype)
            
            protein_expressions = torch.stack(protein_expressions)
        
        # Ensure correct device and dtype
        if protein_expressions.device != self.device:
            protein_expressions = protein_expressions.to(self.device)
        
        if protein_expressions.dtype != self.dtype:
            protein_expressions = protein_expressions.to(self.dtype)
        
        # Forward through MLP encoder
        with torch.set_grad_enabled(self.training):
            protein_features = self.mlp_encoder(protein_expressions)
        
        return protein_features
    
    @property
    def dummy_feature(self):
        """Dummy feature for compatibility with TinyLLaVA"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        """Get model dtype"""
        if self.is_loaded and self.mlp_encoder is not None:
            return next(self.mlp_encoder.parameters()).dtype
        return torch.float32
    
    @property 
    def device(self):
        """Get model device"""
        if self.is_loaded and self.mlp_encoder is not None:
            return next(self.mlp_encoder.parameters()).device
        return torch.device('cpu')
    
    @property
    def config(self):
        """Configuration for compatibility"""
        return {
            'hidden_size': self.hidden_size,
            'num_proteins': self.num_proteins,
            'mlp_tower_name': self.mlp_tower_name,
            'dropout': self.dropout
        }
    
    def freeze(self):
        """Freeze MLP parameters"""
        if self.is_loaded:
            for param in self.mlp_encoder.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze MLP parameters"""
        if self.is_loaded:
            for param in self.mlp_encoder.parameters():
                param.requires_grad = True
    
    def get_num_params(self):
        """Get number of parameters"""
        if self.is_loaded:
            return sum(p.numel() for p in self.mlp_encoder.parameters())
        return 0
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        if self.is_loaded:
            return sum(p.numel() for p in self.mlp_encoder.parameters() if p.requires_grad)
        return 0


def build_mlp_tower(mlp_tower_cfg, **kwargs):
    """Builder function for MLP tower (following TinyLLaVA pattern)"""
    return MLPTower(mlp_tower_cfg, **kwargs)


# For compatibility with different loading patterns
def create_mlp_tower(config_dict):
    """Alternative creation function"""
    return MLPTower(config_dict)