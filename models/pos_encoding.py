import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 5000):
        """
        emb_dim : 与 Transformer d_model 一致
        max_len : 能处理的最大序列长度
        """
        super().__init__()
        # 生成一个 [max_len, emb_dim] 的位置编码矩阵
        position_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2, dtype=torch.float) *
            -(math.log(10000.0) / emb_dim)
        )  # [emb_dim/2]
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，不参与梯度更新，但会随着模型 .to(device) 移动
        self.register_buffer('pe', position_encoding.unsqueeze(0))  # [1, max_len, emb_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, emb_dim]
        返回 x + positional_encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]