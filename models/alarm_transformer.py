import torch
from torch import nn
from models.pos_encoding import PositionalEncoding

class AlarmTransformer(nn.Module):
    def __init__(self, input_dim, emb_dim, nhead, hid_dim, max_len, nlayers, dropout):
        """
        初始化模型各层
        参数:
            input_dim: 输入特征维度，例如文本特征长度
            emb_dim:   Transformer 内部使用的 embedding 维度
            nhead:     多头注意力的头数
            hid_dim:   前馈网络隐藏层维度
            max_len:   序列的最大长度
            nlayers:   TransformerEncoder 的层数
            dropout:   dropout 概率
        """
        super().__init__()
        # 1) 将输入投影到 emb_dim
        self.input_proj = nn.Linear(input_dim, emb_dim)
        # 2) 位置编码
        self.pos_enc = PositionalEncoding(emb_dim, max_len)
        # 3) 多层 TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        # 4) 池化 & 归一化
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_seq: bool = False):
        """
        前向计算
        参数:
            x: 输入张量 ``[batch_size, seq_len, input_dim]``

        返回:
            ``[batch_size, emb_dim]`` 的序列表示向量
        """
        # 投影到 embedding 空间
        embedded = self.input_proj(x)             # [B, L, emb_dim]
        # 加入位置编码
        embedded = self.pos_enc(embedded)         # [B, L, emb_dim]
        # Transformer 编码
        transformed = self.transformer(embedded)  # [B, L, emb_dim]
        # 自适应平均池化到一个向量
        trans_perm = transformed.permute(0, 2, 1)
        pooled = self.pool(trans_perm).squeeze(-1)  # [B, emb_dim]
        # 归一化并 dropout
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        if return_seq:
            return pooled, transformed
        return pooled