import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GINEConv
from torch_geometric.nn.models import MLP
from torch.nn import MultiheadAttention


class GNNTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 输入映射：将原始节点特征(3维)映射到 hidden_channels
        self.lin_dict = nn.ModuleDict({
            ntype: nn.Linear(in_channels, hidden_channels)
            for ntype in ['core', 'agg', 'access']
        })

        # 跨模态注意力机制
        self.attention = MultiheadAttention(embed_dim=hidden_channels, num_heads=4, dropout=dropout)

        # 构建多层 HeteroConv，每层内部用 GINEConv(edge_dim=3)
        self.convs = nn.ModuleList()
        rels = [
            ('core','to','core'),
            ('core','to','agg'),
            ('agg','to','core'),
            ('agg','to','agg'),
            ('agg','to','access'),
            ('access','to','agg'),
            ('access','to','access'),
        ]
        for _ in range(num_layers):
            conv = HeteroConv({
                rel: GINEConv(
                    nn=MLP([hidden_channels, hidden_channels, hidden_channels]),
                    edge_dim=3    # ← 这里一定要加上
                )
                for rel in rels
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1) 映射到同一维度
        h_dict = {ntype: F.elu(self.lin_dict[ntype](x))
                  for ntype, x in x_dict.items()}

        # 2) 多层卷积 + 残差
        for conv in self.convs:
            h_new = conv(h_dict, edge_index_dict, edge_attr_dict)
            for ntype in h_dict:
                h = F.elu(h_new[ntype] + h_dict[ntype])
                h = F.dropout(h, p=self.dropout, training=self.training)
                h_dict[ntype] = h

        return h_dict
