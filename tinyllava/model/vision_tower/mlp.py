from . import register_vision_tower
from .base import VisionTower

@register_vision_tower('mlp')      
class MLPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        from ..biomolecule_multimodal_encoder.mlp_tower import MLPTower
        
        # Create MLPTower with config parameters
        mlp_config = {
            'mm_mlp_tower': getattr(cfg, 'mlp_tower_type', 'mlp_3'),
            'num_proteins': getattr(cfg, 'num_proteins', 4792),
            'hidden_size': getattr(cfg, 'hidden_size', 256),
            'dropout': getattr(cfg, 'dropout', 0.3)
        }
        
        self._vision_tower = MLPTower(mlp_config, delay_load=False)
        self._image_processor = None  # No image processor needed for proteomics

    def forward(self, x, **kwargs):
        """Forward pass for proteomics data"""
        # x should be proteomics tensor of shape [batch_size, num_proteins]
        proteomics_features = self._vision_tower(x)
        
        # Add a sequence dimension to match expected format [batch_size, seq_len, hidden_size]
        # For proteomics, we treat it as a single "token" representing the entire profile
        if len(proteomics_features.shape) == 2:
            proteomics_features = proteomics_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        return proteomics_features
    
    def load_model(self, vision_tower_name, **kwargs):
        print(f"MLP tower already initialized in __init__: {vision_tower_name}")

    def _load_model(self, vision_tower_name, **kwargs):
        """Override the base class load method since MLP doesn't need pretrained weights"""
        # MLPTower doesn't need to load pretrained weights like vision transformers
        # The MLP will be randomly initialized and trained from scratch
        print(f"Initializing MLP tower: {vision_tower_name}")
        # No need to call from_pretrained() like other vision towers
        pass

    @property
    def hidden_size(self):
        """Return the hidden size of the MLP tower"""
        return self._vision_tower.hidden_size if self._vision_tower else 256

    @property
    def num_patches(self):
        """For compatibility - proteomics has 1 'patch' (the entire profile)"""
        return 1