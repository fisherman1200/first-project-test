import torch
from torch import nn
from models.pos_encoding import PositionalEncoding, RotaryXPos, apply_rope_like
from torch.nn.attention.flex_attention import flex_attention as _flex

NEG_INF = -1e30

class AlarmTransformer(nn.Module):
    def __init__(self, input_dim, emb_dim, nhead, hid_dim, max_len, nlayers, dropout,
                  return_seq=False ,use_rope=True):
        super().__init__()
        self.return_seq = return_seq
        self.input_proj = nn.Linear(input_dim, emb_dim)

        if use_rope:
            # 不再使用加法式的位置编码
            self.encoder_layers = nn.ModuleList([
                RotaryEncoderLayer(emb_dim, nhead, hid_dim, dropout=dropout)
                for _ in range(nlayers)
            ])
            self.use_rope = True
        else:
            # 沿用原有：正弦/余弦位置编码 + nn.TransformerEncoder
            self.pos_encoder = PositionalEncoding(emb_dim, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=nhead, dim_feedforward=hid_dim,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
            self.use_rope = False

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, return_seq=None):
        # x: [B,L,input_dim]
        return_seq = self.return_seq if return_seq is None else return_seq
        x = self.input_proj(x)  # [B,L,d]

        if self.use_rope:
            h = x
            for layer in self.encoder_layers:
                h = layer(h, key_padding_mask=key_padding_mask)  # [B,L,d]
        else:
            h = self.pos_encoder(x)
            h = self.transformer(h, src_key_padding_mask=key_padding_mask)  # [B,L,d]

        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # [B,d]
        pooled = self.drop(self.norm(pooled))
        return (pooled, h) if return_seq else pooled


class XPosFlexMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 rope_base=10000.0, max_pos=4096, xpos_scale_base=512.0,
                 use_causal=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        # XPos：scale_base=None 时即 RoPE
        self.rope = RotaryXPos(self.head_dim, base=rope_base, max_pos=max_pos,
                               scale_base=xpos_scale_base)
        self.dropout = dropout
        self.use_causal = use_causal

        try:
            # fullgraph=True 通常更稳；遇到动态图可去掉或加 dynamic=True
            self.flex = torch.compile(_flex, fullgraph=True)
            self._flex_compiled = True
        except Exception:
            # 降级为未编译版本（会看到你之前的 warning，但能跑）
            self.flex = _flex
            self._flex_compiled = False

    def forward(self, x, key_padding_mask=None,
                time_ids=None, topo_bias=None,  # 先验：时间/拓扑，可选
                decay_alpha: float = 0.0):      # 时间衰减强度，0 表示不用
        """
        x: [B,L,D], key_padding_mask: [B,L] (True=有效 or 1=有效)
        time_ids: [B,L]  时间戳或相对步，可同时用于多样本（整型/浮点皆可）
        topo_bias: [L,L] 或 [B,L,L]  先验偏置（例如来自拓扑最短路），数值会直接加到 logits
        """
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, L, H, Dh).permute(0,2,1,3)  # [B,H,L,Dh]
        k = self.k_proj(x).view(B, L, H, Dh).permute(0,2,1,3)
        v = self.v_proj(x).view(B, L, H, Dh).permute(0,2,1,3)

        # —— 先 XPos 旋转 Q/K ——
        cos_q, sin_q, cos_k, sin_k = self.rope.get_cos_sin(L, device=x.device, dtype=x.dtype)
        q, k = apply_rope_like(q, k, cos_q, sin_q, cos_k, sin_k)     # [B,H,L,Dh]

        # —— FlexAttention：定义可编程打分 —— #
        # 这里的 score_mod 会在每个 (b,h,q_idx,k_idx) 对上被调用，数值直接加到缩放后的 QK^T 上
        # 你可以在这里写入任意先验（时间衰减、拓扑偏置、padding/因果遮罩）
        kp = None
        if key_padding_mask is not None:
            # 规范化为 bool，True=有效（可参与注意力）
            kp = key_padding_mask.to(dtype=torch.bool)

        # 预处理：把时间与拓扑偏置 broadcast 成闭包可索引对象
        if time_ids is not None:
            # 确保为 float，形如 [B,L]
            time_ids = time_ids.to(dtype=torch.float32)
        if topo_bias is not None and topo_bias.dim() == 2:
            topo_bias = topo_bias.unsqueeze(0).expand(B, -1, -1)     # [B,L,L]

        def score_mod(score, b, h, q_idx, k_idx):
            zero = score.new_zeros(())
            neg = torch.tensor(-1e9, device=score.device, dtype=score.dtype)
            bias = zero
            if self.use_causal:  # 布尔常量可以 if
                bias = bias + torch.where(q_idx < k_idx, neg, zero)
            if kp is not None:  # None 判断可以 if
                bias = bias + torch.where(kp[b, k_idx], zero, neg)  # 注意语义
            if time_ids is not None and decay_alpha > 0:
                dt = (time_ids[b, q_idx] - time_ids[b, k_idx]).abs().to(score.dtype)
                bias = bias - decay_alpha * dt
            if topo_bias is not None:
                bias = bias + topo_bias[b, q_idx, k_idx].to(score.dtype)
            return score + bias

        # 1) 生成 cos/sin 时，用 q 的 dtype（更稳）
        cos_q, sin_q, cos_k, sin_k = self.rope.get_cos_sin(L, device=x.device, dtype=q.dtype)
        q, k = apply_rope_like(q, k, cos_q, sin_q, cos_k, sin_k)

        # 2) 统一 Q/K/V 的 dtype（以 v.dtype 为准，配合 AMP 更高效）
        dt = v.dtype
        if q.dtype != dt: q = q.to(dt)
        if k.dtype != dt: k = k.to(dt)
        # v 本来就是 dt，可不转


        # === 调用 FlexAttention，并兼容不同返回形式 ===
        res = self.flex(q, k, v, score_mod=score_mod)  # 可能返回 Tensor 或 (out, lse)

        if isinstance(res, tuple):
            ctx, _lse = res
        else:
            ctx = res

        if ctx is None:
            raise RuntimeError("flex_attention returned None; "
                               "check score_mod logic and torch.compile path.")

        # [B,H,L,Dh] -> [B,L,D]
        out = ctx.permute(0, 2, 1, 3).contiguous().view(B, L, D)

        # 一定要返回张量
        return self.o_proj(out)



class RotaryEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 use_causal=True, xpos_scale_base=4096):
        super().__init__()
        self.self_attn = XPosFlexMHA(
            d_model, nhead, dropout=dropout,
            rope_base=10000.0, max_pos=4096,
            xpos_scale_base=xpos_scale_base,   # =None 时退化为 RoPE
            use_causal=use_causal
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model);  self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, src, key_padding_mask=None, time_ids=None, topo_bias=None, decay_alpha=0.0):
        # src: [B,L,D]
        x = src
        x = x + self.dropout1(self.self_attn(x, key_padding_mask, time_ids, topo_bias, decay_alpha))
        x = self.norm1(x)
        x = x + self.dropout2(self.linear2(self.act(self.linear1(x))))
        x = self.norm2(x)
        return x