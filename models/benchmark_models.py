import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphConv
from transformers import AutoModel
import os


"""本文件收录若干用于性能对比的模型实现，
包括 CONAD、LogBERT、LogGD、DeepTraLog、Graphormer、
GraphMAE 以及 DistilBERTGraph 多模态模型。
实现均为简化示例，方便在本项目中快速实验。
"""

class CONAD(nn.Module):
    """基于 LSTM 的简化版 CONAD 模型

    主要用于处理告警日志序列，通过 LSTM 提取时序特征，
    最终输出二分类结果。
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # 兼容输入形状为 [batch, feat_dim] 的情况，
        # 此时视为序列长度为 1
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # x 形状: [batch, seq_len, feat_dim]
        # h[-1] 形状: [batch, hidden_dim]
        _, (h, _) = self.lstm(x)
        out = self.classifier(h[-1])
        return out

class LogBERT(nn.Module):
    """使用预训练 BERT 的日志表示模型

    数据集中已经离线提取好了文本表示及其他特征，本实现
    直接对这些特征进行两层感知机分类，避免再次调用 BERT。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, attn_mask=None):
        return self.mlp(x)


class LogGD(nn.Module):
    """基于图卷积的日志表示模型
    使用两层 ``GraphConv`` 提取节点间的结构信息，
    然后平均池化得到图级表示。模型内部会根据 ``edge_index``
    自动处理批量维度。输入 ``x`` 为 ``[B, F]`` 的序列级特征。
    """
    def __init__(self, in_channels, edge_index, hidden_channels=64):
        super().__init__()
        # 保存图结构并预计算节点数，方便批量展开
        self.register_buffer("edge_index", edge_index)
        self.num_nodes = int(edge_index.max()) + 1
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def _repeat_graph(self, x):
        """将批量特征展开为 ``batch_size*num_nodes`` 个节点"""
        bsz = x.size(0)
        # [B, F] -> [B, N, F] -> [B*N, F]
        x_rep = x.unsqueeze(1).expand(-1, self.num_nodes, -1).reshape(-1, x.size(1))

        # 边索引也按批量复制，并在不同批次间加上偏移量
        offset = torch.arange(bsz, device=x.device).view(bsz, 1, 1) * self.num_nodes
        edge = self.edge_index.unsqueeze(0) + offset
        edge = edge.reshape(2, -1)
        return x_rep, edge

    def forward(self, x):
        """前向传播，``x`` 形状为 ``[batch, feat_dim]``"""
        x_rep, edge = self._repeat_graph(x)
        h = torch.relu(self.conv1(x_rep, edge))
        h = torch.relu(self.conv2(h, edge))
        h = h.view(x.size(0), self.num_nodes, -1).mean(dim=1)
        # 图级向量经过线性层得到二分类结果
        return self.classifier(h)

class DeepTraLog(nn.Module):
    """使用注意力的日志序列模型

    先通过 LSTM 抽取序列特征，随后利用
    MultiheadAttention 进一步建模长程依赖。
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # 当输入为 [B, D] 时说明已经做过池化，此处补上长度维度
        if x.dim() == 2:
            x = x.unsqueeze(1)

        h, _ = self.lstm(x)
        attn_out, _ = self.attn(h, h, h)
        pooled = attn_out.mean(dim=1)
        return self.classifier(pooled)

class Graphormer(nn.Module):
    """简化版 Graphormer

    通过多层 TransformerEncoder 建模图结构，
    这里只演示核心思想，并非完整实现。
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_channels, 4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x):
        if x.dim() == 2:
            # [B, D] -> [B, 1, D]
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)

class GraphMAE(nn.Module):
    """图自编码器，用于无监督预训练

    输入图经过编码得到隐藏向量，再重构回原特征，
    可在大规模数据上学习通用表示。
    """
    def __init__(self, in_channels, edge_index, hidden_channels=64):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.num_nodes = int(edge_index.max()) + 1
        self.encoder = GraphConv(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, in_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def _repeat_graph(self, x):
        bsz = x.size(0)
        # [B, F] -> [B, N, F] -> [B*N, F]
        x_rep = x.unsqueeze(1).expand(-1, self.num_nodes, -1).reshape(-1, x.size(1))
        # 复制边索引并添加批量偏移量
        offset = torch.arange(bsz, device=x.device).view(bsz, 1, 1) * self.num_nodes
        edge = self.edge_index.unsqueeze(0) + offset
        edge = edge.reshape(2, -1)
        return x_rep, edge

    def forward(self, x):
        """计算并返回二分类 logits"""
        x_rep, edge = self._repeat_graph(x)
        h = torch.relu(self.encoder(x_rep, edge))
        # [B*N, H] -> [B, N, H] -> [B, H]
        h = h.view(x.size(0), self.num_nodes, -1).mean(dim=1)
        return self.classifier(h)

class DistilBERTGraph(nn.Module):
    """DistilBERT 与 GNN 融合的多模态模型

    文本部分使用 DistilBERT 编码，图部分使用 GAT，
    将两种特征拼接后进行分类。
    """
    def __init__(self, in_channels, edge_index, text_model: str = "distilbert-base-uncased"):
        super().__init__()
        # 如果设置了离线环境变量，则只从本地缓存读取模型，不尝试联网下载
        offline = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"

        try:
            self.bert = AutoModel.from_pretrained(text_model, local_files_only=offline)
        except Exception as e:
            # 加载失败通常是因为无法联网或本地缓存不存在
            msg = (
                "无法加载预训练的 DistilBERT 模型，请确保网络可用，"
                "或提前下载模型并设置 TRANSFORMERS_OFFLINE=1。"
            )
            raise RuntimeError(msg) from e

        # 文本向量维度
        self.text_dim = self.bert.config.hidden_size
        # 计算 GNN 输入维度：总维度减去文本维度
        self.gnn_in_dim = max(in_channels - self.text_dim, 1)

        # 图结构常量在初始化时传入并保存
        self.register_buffer("edge_index", edge_index)
        self.num_nodes = int(edge_index.max()) + 1

        self.gnn = GATConv(self.gnn_in_dim, self.gnn_in_dim)
        self.fc = nn.Linear(self.text_dim + self.gnn_in_dim, 2)

    def _repeat_graph(self, x):
        """把批量特征扩展成整图输入"""
        bsz = x.size(0)
        x_rep = x.unsqueeze(1).expand(-1, self.num_nodes, -1).reshape(-1, x.size(1))
        offset = torch.arange(bsz, device=x.device).view(bsz, 1, 1) * self.num_nodes
        edge = self.edge_index.unsqueeze(0) + offset
        edge = edge.reshape(2, -1)
        return x_rep, edge

    def forward(self, x):
        """计算并返回二分类 logits"""
        # 拆分文本与图特征
        text_feat = x[:, : self.text_dim]
        gnn_x = x[:, self.text_dim:]

        x_rep, edge = self._repeat_graph(gnn_x)
        gnn_feat = torch.relu(self.gnn(x_rep, edge))
        gnn_feat = gnn_feat.view(x.size(0), self.num_nodes, -1).mean(dim=1)
        fused = torch.cat([text_feat, gnn_feat], dim=-1)
        return self.fc(fused)