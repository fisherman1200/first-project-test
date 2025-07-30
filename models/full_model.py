import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torchinfo import summary

from models.gnn_transformer import GNNTransformer
from models.alarm_transformer import AlarmTransformer
from utils.config import load_config


class FullModel(nn.Module):
    """融合 GNN 与 Transformer 的完整网络模型"""

    def __init__(self, cfg, feature_dim, input_dim):
        super().__init__()
        # GNN 模块
        self.gnn = GNNTransformer(
            in_channels=feature_dim,
            hidden_channels=cfg.gnn.hidden_channels,
            num_layers=cfg.gnn.num_layers,
            dropout=cfg.gnn.dropout,
        )

        # 文本 Transformer 模块
        self.alarm_transformer = AlarmTransformer(
            input_dim=input_dim,
            emb_dim=cfg.transformer.emb_dim,
            nhead=cfg.transformer.nhead,
            hid_dim=cfg.transformer.hid_dim,
            nlayers=cfg.transformer.nlayers,
            max_len=cfg.transformer.max_len,
            dropout=cfg.transformer.dropout,
        )

        # 跨模态注意力与门控机制
        self.cross_attention = MultiheadAttention(embed_dim=cfg.transformer.emb_dim, num_heads=4, dropout=0.3)
        # 专门给真实故障判别用的注意力模块
        self.true_attention = MultiheadAttention(embed_dim=cfg.transformer.emb_dim, num_heads=4, dropout=0.1)
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
        # head_true 前添加两层非线性映射，更强的表达能力
        self.head_true = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # 序列级根因定位
        self.token_root_head = nn.Linear(cfg.transformer.emb_dim, 2)

        # 预计算的节点嵌入缓冲区
        self.node_embs = None

    def compute_node_embs(self, x_dict, edge_index_dict, edge_attr_dict):
        """预计算所有节点的嵌入并缓存"""
        with torch.no_grad():
            h_dict = self.gnn(x_dict, edge_index_dict, edge_attr_dict)
            h_core = h_dict['core']
            h_agg = h_dict['agg']
            h_access = h_dict['access']
            pad = torch.zeros(1, h_core.size(1), dtype=h_core.dtype, device=h_core.device)
            embs = torch.cat([pad, h_core, h_agg, h_access], dim=0)
        self.node_embs = embs.detach()
        return self.node_embs

    def forward(self, batch):
        """执行前向计算，返回根告警与真故障预测"""
        if self.node_embs is None:
            raise RuntimeError("请先调用 compute_node_embs 计算节点嵌入")
        # 1. 节点嵌入平均池化
        seq_node_embs = self.node_embs[batch['node_idxs']]  # [batch_size, max_len, hidden_channels]
        node_feat = seq_node_embs.mean(dim=1) # [batch_size, hidden_channels]
        # 2. Transformer 处理告警序列
        pooled_text, seq_feat = self.alarm_transformer(batch['text_feat'], return_seq=True) # [batch_size, emb_dim]
        token_logits = self.token_root_head(seq_feat) # [batch_size, max_len, 2]
        # 3. 跨模态注意力
        # 构造序列：2 tokens × B samples × 64 dim (hidden_channels = emb_dim = 64)
        seq = torch.stack([node_feat, pooled_text], dim=0)  # [2, B, 64]
        attn_out, _ = self.cross_attention(seq, seq, seq)           # [2, B, 64]
        # 取两 token 的平均作为跨模态输出
        attn_fused = attn_out.mean(dim=0)   # [B, 64]
        # 4. 门控融合
        concat_features = torch.cat([node_feat, pooled_text], dim=-1)   # [B, 128]
        gate = self.gating_net(concat_features)                                 # [B, 1] in (0,1)
        fused = gate * attn_fused + (1 - gate) * pooled_text                    # [B, 64]
        # 5. 根告警分类分支
        root_feat = self.shared_root(fused)  # [B, 128]
        out_root = self.head_root(root_feat)  # [B, 2]
        # 6. 真故障分类分支，加入专门的注意力机制
        seq_perm = seq_feat.permute(1, 0, 2)  # [L, B, 64]
        attn_out, _ = self.true_attention(fused.unsqueeze(0), seq_perm, seq_perm)
        true_context = attn_out.squeeze(0)  # [B, 64]
        true_input = torch.cat([fused, true_context], dim=-1)  # [B, 128]
        true_feat = self.shared_true(true_input)
        out_true = self.head_true(true_feat)

        return out_root, out_true, token_logits

if __name__ == "__main__":
    cfg = load_config("../configs/config.yaml")
    model = FullModel(cfg, 3, 834) # 768 + 2 + 32 + 32
    print("===== 整体网络结构 =====")
    summary(model)