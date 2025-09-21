from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('node_encoder')      
class NodeVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        from ..biomolecule_multimodal_encoder.node_tower import NodeTower
        
        # Create node tower with config parameters
        node_config = {
            'node_tower_type': getattr(cfg, 'node_tower_type', 'gcn'),
            'num_proteins': getattr(cfg, 'num_proteins', 4792),
            'hidden_size': getattr(cfg, 'hidden_size', 512),
            'dropout': getattr(cfg, 'dropout', 0.3),
            'k_neighbors': getattr(cfg, 'k_neighbors', 7),
            'proteomics_data_path': getattr(cfg, 'proteomics_data_path', None)
        }
        
        self._vision_tower = NodeTower(node_config, delay_load=False)
        self._image_processor = None

    def forward(self, sample_ids, **kwargs):
        # sample_ids should be the proteomics sample identifiers
        node_embeddings = self._vision_tower(sample_ids)
        
        # Add sequence dimension to match expected format [batch_size, seq_len, hidden_size]
        if len(node_embeddings.shape) == 2:
            node_embeddings = node_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        return node_embeddings

    def _load_model(self, vision_tower_name, **kwargs):
        # Node tower handles its own initialization
        print(f"Initializing node tower: {vision_tower_name}")
        pass

    @property
    def hidden_size(self):
        return self._vision_tower.hidden_size if self._vision_tower else 512

    @property
    def num_patches(self):
        return 1