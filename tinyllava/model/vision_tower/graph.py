from . import register_vision_tower
from .base import VisionTower

@register_vision_tower('graph_tower')      
class GraphVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        from ..biomolecule_multimodal_encoder.graph_tower import GraphTower
        
        graph_config = {
            'graph_tower_type': getattr(cfg, 'graph_tower_type', 'gcn'),
            'hidden_size': getattr(cfg, 'hidden_size', 512),
            'dropout': getattr(cfg, 'dropout', 0.3),
            'patient_graphs_dir': getattr(cfg, 'patient_graphs_dir', None)
        }
        
        self._vision_tower = GraphTower(graph_config, delay_load=False)
        self._image_processor = None

    def forward(self, sample_ids, **kwargs):
        graph_embeddings = self._vision_tower(sample_ids)
        if len(graph_embeddings.shape) == 2:
            graph_embeddings = graph_embeddings.unsqueeze(1)
        return graph_embeddings

    def load_model(self, vision_tower_name, **kwargs):
        print(f"GraphTower already initialized in __init__: {vision_tower_name}")


    def _load_model(self, vision_tower_name, **kwargs):
        print(f"Initializing graph tower: {vision_tower_name}")
        pass

    @property
    def hidden_size(self):
        return self._vision_tower.hidden_size

    @property
    def num_patches(self):
        return 1