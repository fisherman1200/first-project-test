import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from models.gnn_transformer import GNNTransformer

class GraphormerLayer(nn.Module):
    """简化版 Graphormer 层：带注意力偏置"""
    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        assert hidden_dim % nhead == 0, "hidden_dim 必须能被 nhead 整除"
        # 自定义 qkv 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias):
        B, N, C = x.size()
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 将预计算的注意力偏置加到权重上
        attn_scores = attn_scores + attn_bias.unsqueeze(0)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)  # [B, nhead, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)

        x = x + out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

class HeteroGraphormer(nn.Module):
    """先用 HeteroGNN 提取节点嵌入，再以 Graphormer 方式编码"""
    def __init__(self, in_channels, hidden_dim, num_layers=3,
                 nhead=8, max_dist=5, dropout=0.1):
        super().__init__()
        self.gnn = GNNTransformer(in_channels, hidden_dim, num_layers, dropout)
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.max_dist = max_dist
        self.spatial_emb = nn.Embedding(max_dist + 1, nhead)
        self.in_deg_emb = nn.Embedding(32, hidden_dim)
        self.out_deg_emb = nn.Embedding(32, hidden_dim)

    @staticmethod
    def _build_graph(edge_index_dict, num_nodes):
        g = nx.DiGraph()
        g.add_nodes_from(range(num_nodes))
        for edges in edge_index_dict.values():
            src, tgt = edges
            edge_list = list(zip(src.tolist(), tgt.tolist()))
            g.add_edges_from(edge_list)
        return g

    def _compute_bias(self, graph):
        N = graph.number_of_nodes()
        # 计算最短路径，限制最大距离
        dist = [[self.max_dist for _ in range(N)] for _ in range(N)]
        for i in range(N):
            dist[i][i] = 0
        path_len = dict(nx.all_pairs_shortest_path_length(graph, cutoff=self.max_dist))
        for i, d in path_len.items():
            for j, l in d.items():
                dist[i][j] = min(l, self.max_dist)
        dist = torch.tensor(dist, dtype=torch.long, device=self.spatial_emb.weight.device)
        bias = self.spatial_emb(dist)  # [N, N, nhead]
        bias = bias.permute(2, 0, 1)   # [nhead, N, N]
        return bias

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1) 先通过 HeteroGNN 获取各类型节点的表示
        feat_dict = self.gnn(x_dict, edge_index_dict, edge_attr_dict)
        # 拼接成一个整体序列：顺序 core -> agg -> access
        feats = torch.cat([
            feat_dict['core'],
            feat_dict['agg'],
            feat_dict['access'],
        ], dim=0)  # [N, hidden_dim]
        N = feats.size(0)

        graph = self._build_graph(edge_index_dict, N)
        bias = self._compute_bias(graph)  # [nhead, N, N]

        # 计算入度/出度编码
        device = self.in_deg_emb.weight.device
        in_deg = torch.tensor([graph.in_degree(i) for i in range(N)],
                              dtype=torch.long, device=device).clamp(max=31)
        out_deg = torch.tensor([graph.out_degree(i) for i in range(N)],
                               dtype=torch.long, device=device).clamp(max=31)
        feats = feats + self.in_deg_emb(in_deg) + self.out_deg_emb(out_deg)

        h = feats.unsqueeze(0)  # [1, N, hidden_dim]
        for layer in self.layers:
            h = layer(h, bias)
        return h.squeeze(0)