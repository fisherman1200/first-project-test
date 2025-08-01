import torch
from models.gnn_transformer import GNNTransformer
from models.base_full_model import BaseFullModel

class FullModel(BaseFullModel):
    """结合 GNN 与文本 Transformer 的整体模型"""

    def __init__(self, cfg, feature_dim, input_dim):
        # 先构建 GNN 模块
        self.gnn = GNNTransformer(
            in_channels=feature_dim,
            hidden_channels=cfg.gnn.hidden_channels,
            num_layers=cfg.gnn.num_layers,
            dropout=cfg.gnn.dropout,
        )
        super().__init__(cfg, feature_dim, input_dim)

    def compute_node_embs(self, x_dict, edge_index_dict, edge_attr_dict):
        """预计算所有节点嵌入并缓存"""
        with torch.no_grad():
            h_dict = self.gnn(x_dict, edge_index_dict, edge_attr_dict)
            h_core = h_dict['core']
            h_agg = h_dict['agg']
            h_access = h_dict['access']
            pad = torch.zeros(1, h_core.size(1), dtype=h_core.dtype, device=h_core.device)
            embs = torch.cat([pad, h_core, h_agg, h_access], dim=0)
        self.node_embs = embs.detach()
        return self.node_embs