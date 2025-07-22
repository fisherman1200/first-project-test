import torch
from torch import nn
from models.pos_encoding import PositionalEncoding

class AlarmTransformer(nn.Module):
    def __init__(self, input_dim, emb_dim, nhead, hid_dim, max_len, nlayers):
        """
        input_dim : text_feat 的维度
        emb_dim   : Transformer embedding 维度
        nhead     : 多头注意力头数
        hid_dim   : feedforward network 中间层维度
        nlayers   : TransformerEncoder 层数
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
            dropout=0.1,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        # 4) 池化 & 归一化
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, input_dim]
        输出: [B, emb_dim]
        """
        # 投影
        h = self.input_proj(x)           # [B, L, emb_dim]
        # 加位置编码
        h = self.pos_enc(h)              # [B, L, emb_dim]
        # 经过 Transformer
        h = self.transformer(h)          # [B, L, emb_dim]
        # 池化: 先调到 [B, emb_dim, L]
        h = h.permute(0, 2, 1)
        v = self.pool(h).squeeze(-1)     # [B, emb_dim]
        # 归一化 + dropout
        v = self.norm(v)
        return self.dropout(v)