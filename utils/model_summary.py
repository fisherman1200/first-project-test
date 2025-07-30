"""打印整体神经网络结构信息的脚本。

本脚本根据 ``configs/config.yaml`` 中的配置，构建与训练流程一致的模型，
并使用 ``torchinfo.summary`` 按照从 GNN 到 Transformer，再到跨模态
融合与多任务头的顺序输出各模块结构。
"""

import torch
import torch.nn as nn
from torchinfo import summary
from models.gnn_transformer import GNNTransformer
from models.alarm_transformer import AlarmTransformer
from utils.config import Config, load_config


class FullModel(nn.Module):
    """训练流程中用到的完整网络。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # 1) GNN 模块
        self.gnn = GNNTransformer(
            in_channels=3,
            hidden_channels=cfg.gnn.hidden_channels,
            num_layers=cfg.gnn.num_layers,
            dropout=cfg.gnn.dropout,
        )

        # 2) Alarm Transformer
        self.alarm_transformer = AlarmTransformer(
            input_dim=834,  # 768 + 2 + 32 + 32
            emb_dim=cfg.transformer.emb_dim,
            nhead=cfg.transformer.nhead,
            hid_dim=cfg.transformer.hid_dim,
            nlayers=cfg.transformer.nlayers,
            max_len=cfg.transformer.max_len,
            dropout=cfg.transformer.dropout,
        )

        # 3) Cross-Attention 与门控机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cfg.transformer.emb_dim,
            num_heads=4,
            dropout=0.3,
        )
        self.gating_net = nn.Sequential(
            nn.Linear(cfg.transformer.emb_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 4) 多任务分类头
        self.shared = nn.Sequential(
            nn.Linear(cfg.transformer.emb_dim, 128),
            nn.ReLU(),
        )
        self.head_root = nn.Linear(128, 2)
        self.head_true = nn.Linear(128, 2)
        self.token_root_head = nn.Linear(cfg.transformer.emb_dim, 2)

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        node_idxs,
        text_feat,
    ):
        """按照训练流程执行一次前向传播。"""

        # ----- GNN -----
        h_dict = self.gnn(x_dict, edge_index_dict, edge_attr_dict)
        pad = torch.zeros(1, h_dict["core"].size(1), device=h_dict["core"].device)
        node_embs = torch.cat([pad, h_dict["core"], h_dict["agg"], h_dict["access"]])
        seq_node_embs = node_embs[node_idxs]
        node_feat = seq_node_embs.mean(dim=1)

        # ----- Alarm Transformer -----
        pooled_text, seq_feat = self.alarm_transformer(text_feat, return_seq=True)
        token_logits = self.token_root_head(seq_feat)

        # ----- Cross-Attention -----
        seq = torch.stack([node_feat, pooled_text], dim=0)
        attn_out, _ = self.cross_attention(seq, seq, seq)
        attn_fused = attn_out.mean(dim=0)

        # ----- Gating Mechanism -----
        concat_feat = torch.cat([node_feat, pooled_text], dim=-1)
        gate = self.gating_net(concat_feat)
        fused = gate * attn_fused + (1 - gate) * pooled_text

        # ----- Multi-Task Head -----
        shared_feat = self.shared(fused)
        out_root = self.head_root(shared_feat)
        out_true = self.head_true(shared_feat)
        return out_root, out_true, token_logits


def print_full_model_info(cfg: Config) -> None:
    """构建 ``FullModel`` 并打印各模块结构信息。"""

    model = FullModel(cfg)

    print("===== 整体网络结构 =====")
    summary(model)


if __name__ == "__main__":
    cfg = load_config("../configs/config.yaml")
    print_full_model_info(cfg)