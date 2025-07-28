import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GINEConv
from torch_geometric.nn.models import MLP
from torch.nn import MultiheadAttention


class GNNTransformer(nn.Module):
    """异构图卷积网络，输出各类型节点的表示。"""
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.3):
        """
        构造函数：初始化异构图卷积网络与注意力层。

        参数:
            in_channels: 原始节点特征维度
            hidden_channels: 输出与内部隐藏维度
            num_layers: 卷积层数
            dropout: Dropout 比例
        """
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
        """
        前向计算：对每种节点类型分别进行卷积更新后输出。

        参数:
            x_dict: Dict[node_type, Tensor(N_nodes, in_channels)] 类型到节点特征的映射
            edge_index_dict: Dict[(src,tgt), Tensor(2, E)] 异构边索引
            edge_attr_dict: Dict[(src,tgt), Tensor(E, edge_dim)] 异构边特征
        返回:
            feat_dict: Dict[node_type, Tensor(N_nodes, hidden_channels)] 一个字典，包含各节点类型的表示向量
        """
        # 1) 映射到统一维度
        feat_dict = {ntype: F.elu(self.lin_dict[ntype](x))
                     for ntype, x in x_dict.items()}

        # 2) 多层卷积 + 残差
        for conv in self.convs:
            new_feat = conv(feat_dict, edge_index_dict, edge_attr_dict)
            for ntype in feat_dict:
                updated = F.elu(new_feat[ntype] + feat_dict[ntype])
                updated = F.dropout(updated, p=self.dropout, training=self.training)
                feat_dict[ntype] = updated

        return feat_dict
