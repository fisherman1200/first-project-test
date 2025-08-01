import torch
from models.graphormer_model import HeteroGraphormer
from models.base_full_model import BaseFullModel


class FullModel(BaseFullModel):
    """融合 HeteroGraphormer 与文本 Transformer 的完整模型"""

    def __init__(self, cfg, feature_dim, input_dim):
        # HeteroGraphormer 模块
        self.graphormer = HeteroGraphormer(
            in_channels=feature_dim,
            hidden_dim=cfg.gnn.hidden_channels,
            num_layers=cfg.gnn.num_layers,
            nhead=cfg.transformer.nhead,
            dropout=cfg.gnn.dropout
        )
        # 兼容旧接口，方便加载预训练权重
        self.gnn = self.graphormer.gnn
        super().__init__(cfg, feature_dim, input_dim)

    def compute_node_embs(self, x_dict, edge_index_dict, edge_attr_dict):
        """预计算并缓存节点嵌入"""
        with torch.no_grad():
            embs = self.graphormer(x_dict, edge_index_dict, edge_attr_dict)
        self.node_embs = embs.detach()
        return self.node_embs