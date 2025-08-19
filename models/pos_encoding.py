import math
import torch
import torch.nn.functional as F
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


class RotaryXPos(nn.Module):
    """
    RoPE / XPos 频表与 cos/sin 缓存。
    - 若 scale_base=None → 标准 RoPE
    - 若 scale_base>0   → XPos：对 Q 用 scale，K 用 1/scale
    """
    def __init__(self, head_dim: int, base: float = 10000.0,
                 max_pos: int = 4096, scale_base: float = 512.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = float(base)
        self.max_pos = max_pos
        self.scale_base = scale_base

        # —— 频率：先得到 [L, Dh/2] ——
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_pos, dtype=torch.float32)  # 位置 [0..L-1]
        freqs = torch.outer(t, inv_freq)  # [L, Dh/2]

        # —— 关键：把每个“频率对”复制到相邻的两维，得到 [L, Dh] ——
        # 方式一：stack 再 reshape（更直观）
        emb = torch.stack((freqs, freqs), dim=-1).reshape(max_pos, head_dim)  # [L, Dh]
        cos_full = emb.cos()  # [L, Dh]
        sin_full = emb.sin()  # [L, Dh]

        if scale_base is None:
            # 纯 RoPE：直接注册 [L, Dh]
            self.register_buffer("cos_q", cos_full, persistent=False)
            self.register_buffer("sin_q", sin_full, persistent=False)
            self.register_buffer("cos_k", cos_full, persistent=False)
            self.register_buffer("sin_k", sin_full, persistent=False)
        else:
            # XPos：中心化的指数缩放（实现等价于 TorchScale/HF 常用写法）
            # power = (pos - L//2) / scale_base XPos：先做“对儿”的缩放 [L, Dh/2]，再成对展开到 [L, Dh]
            power = (t - (max_pos // 2)) / float(scale_base)  # [L]
            # 通道系数：随维度递增的 ζ̂_i（这里用常见近似，等价实现见引用）
            # 让 scale 形状 [L, Dh/2]，对每个二维通道成对缩放
            pair_scaler = (torch.arange(0, head_dim, 2).float() + 0.4*head_dim) / (1.4*head_dim)
            scale_pair = pair_scaler.unsqueeze(0) ** power.unsqueeze(1)  # [L, Dh/2]
            scale_full = torch.stack((scale_pair, scale_pair), dim=-1).reshape(max_pos, head_dim)  # [L, Dh]
            self.register_buffer("cos_q", cos_full * scale_full, persistent=False)
            self.register_buffer("sin_q", sin_full * scale_full, persistent=False)
            self.register_buffer("cos_k", cos_full / scale_full, persistent=False)
            self.register_buffer("sin_k", sin_full / scale_full, persistent=False)

    def get_cos_sin(self, seq_len: int, device=None, dtype=None):
        sl = min(seq_len, self.max_pos)
        return (self.cos_q[:sl].to(device=device, dtype=dtype),
                self.sin_q[:sl].to(device=device, dtype=dtype),
                self.cos_k[:sl].to(device=device, dtype=dtype),
                self.sin_k[:sl].to(device=device, dtype=dtype))

def rotate_half(x):
    # (x0,x1)->(-x1,x0) 按偶/奇通道成对旋转
    x1 = x[..., ::2]; x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rope_like(q, k, cos_q, sin_q, cos_k, sin_k):
    """
    q,k: [B, H, L, Dh]
    cos/sin_*: [L, Dh]  会自动广播到 [B,H,L,Dh]
    """
    for t in (cos_q, sin_q, cos_k, sin_k):
        assert t.dim() == 2
    # 扩到 [1,1,L,Dh]
    def _e(a): return a.unsqueeze(0).unsqueeze(0)
    cq, sq, ck, sk = map(_e, (cos_q, sin_q, cos_k, sin_k))
    q = q * cq + rotate_half(q) * sq
    k = k * ck + rotate_half(k) * sk
    return q, k