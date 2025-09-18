import os

from ...utils import import_modules
from .base import VisionTower  # Add this import


VISION_TOWER_FACTORY = {}

def VisionTowerFactory(vision_tower_name):
    vision_tower_name = vision_tower_name.split(':')[0]
    model = None
    for name in VISION_TOWER_FACTORY.keys():
        if name.lower() in vision_tower_name.lower():
            model = VISION_TOWER_FACTORY[name]
    assert model, f"{vision_tower_name} is not registered"
    return model


def register_vision_tower(name):
    def register_vision_tower_cls(cls):
        if name in VISION_TOWER_FACTORY:
            return VISION_TOWER_FACTORY[name]
        VISION_TOWER_FACTORY[name] = cls
        return cls
    return register_vision_tower_cls



# Add this to the imports
from ..biomolecule_multimodal_encoder.mlp_tower import MLPTower

# Add the MLPVisionTower class here
@register_vision_tower('mlp')      
class MLPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        from ..biomolecule_multimodal_encoder.mlp_tower import MLPTower
        self._vision_tower = MLPTower(cfg)
        self._image_processor = None  # No image processor needed for proteomics

    def forward(self, x, **kwargs):
        """Forward pass for proteomics data"""
        # x should be proteomics tensor of shape [batch_size, num_proteins]
        proteomics_features = self._vision_tower(x)
        # Add a sequence dimension to match expected format [batch_size, seq_len, hidden_size]
        if len(proteomics_features.shape) == 2:
            proteomics_features = proteomics_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        return proteomics_features

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.vision_tower")
