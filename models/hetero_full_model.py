import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from models.graphormer_model import HeteroGraphormer
from models.alarm_transformer import AlarmTransformer


class FullModel(nn.Module):
    """融合 HeteroGraphormer 与文本 Transformer 的完整模型"""

    def __init__(self, cfg, feature_dim, input_dim):
        super().__init__()
        # HeteroGraphormer 模块
        self.graphormer = HeteroGraphormer(
            in_channels=feature_dim,
            hidden_dim=cfg.gnn.hidden_channels,
            num_layers=cfg.gnn.num_layers,
            nhead=cfg.transformer.nhead,
            dropout=cfg.gnn.dropout
        )
        # 兼容旧接口，便于预训练
        self.gnn = self.graphormer.gnn

        # 文本 Transformer
        self.alarm_transformer = AlarmTransformer(
            input_dim=input_dim,
            emb_dim=cfg.transformer.emb_dim,
            nhead=cfg.transformer.nhead,
            hid_dim=cfg.transformer.hid_dim,
            nlayers=cfg.transformer.nlayers,
            max_len=cfg.transformer.max_len,
            dropout=cfg.transformer.dropout,
        )

        # 跨模态注意力与门控
        self.cross_attention = MultiheadAttention(cfg.transformer.emb_dim, num_heads=4, dropout=0.3)
        self.true_attention = MultiheadAttention(cfg.transformer.emb_dim, num_heads=4, dropout=0.1)
        self.gating_net = nn.Sequential(
            nn.Linear(cfg.transformer.emb_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 分类头
        self.shared_root = nn.Sequential(nn.Linear(cfg.transformer.emb_dim, 128), nn.ReLU())
        self.shared_true = nn.Sequential(nn.Linear(cfg.transformer.emb_dim * 2, 128), nn.ReLU())
        self.head_root = nn.Linear(128, 2)
        self.head_true = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        # 序列级根因定位
        self.token_root_head = nn.Linear(cfg.transformer.emb_dim, 2)

        # 缓存节点嵌入
        self.node_embs = None

    def compute_node_embs(self, x_dict, edge_index_dict, edge_attr_dict):
        """预计算并缓存节点嵌入"""
        with torch.no_grad():
            embs = self.graphormer(x_dict, edge_index_dict, edge_attr_dict)
        self.node_embs = embs.detach()
        return self.node_embs

    def forward(self, batch):
        """前向计算，返回根告警预测、真实故障预测及每个位置的根因概率"""
        if self.node_embs is None:
            raise RuntimeError("请先调用 compute_node_embs 计算节点嵌入")

        # 1. 节点表示平均池化得到序列表示
        seq_node_embs = self.node_embs[batch['node_idxs']]
        node_feat = seq_node_embs.mean(dim=1)

        # 2. 文本 Transformer 处理告警序列
        pooled_text, seq_feat = self.alarm_transformer(batch['text_feat'], return_seq=True)
        token_logits = self.token_root_head(seq_feat)

        # 3. 跨模态注意力融合
        seq = torch.stack([node_feat, pooled_text], dim=0)
        attn_out, _ = self.cross_attention(seq, seq, seq)
        attn_fused = attn_out.mean(dim=0)

        concat = torch.cat([node_feat, pooled_text], dim=-1)
        gate = self.gating_net(concat)
        fused = gate * attn_fused + (1 - gate) * pooled_text

        # 4. 根告警分类
        root_feat = self.shared_root(fused)
        out_root = self.head_root(root_feat)

        # 5. 真故障分类分支
        seq_perm = seq_feat.permute(1, 0, 2)
        attn_out, _ = self.true_attention(fused.unsqueeze(0), seq_perm, seq_perm)
        true_ctx = attn_out.squeeze(0)
        true_input = torch.cat([fused, true_ctx], dim=-1)
        true_feat = self.shared_true(true_input)
        out_true = self.head_true(true_feat)

        return out_root, out_true, token_logits