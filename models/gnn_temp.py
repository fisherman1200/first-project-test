import torch
from torch import nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GNNTransformer(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, heads=4):
        super().__init__()
        # 用两层 GATConv 代替原先的 NNConv
        # 第一层：in_channels -> hidden_channels，multi‐head attention
        # 两层 GATConv，每层多头（默认 heads=4），输出维度 =(hidden_channels/heads) * heads = hidden_channels。
        self.conv1 = GATConv(
            in_channels, hidden_channels // heads,
            heads=heads,
            concat=True,      # 将每个 head 的输出拼接
            dropout=0.2,
            add_self_loops=True
        )
        # 第二层：hidden_channels -> hidden_channels
        self.conv2 = GATConv(
            hidden_channels, hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=0.2,
            add_self_loops=True
        )

    def forward(self, x, edge_index, edge_attr=None):
        # GATConv 不使用 edge_attr，如果传了也会忽略
        # 激活与 dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return x