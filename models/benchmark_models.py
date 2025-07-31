import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphConv
from transformers import AutoModel

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
        _, (h, _) = self.lstm(x)
        out = self.classifier(h[-1])
        return out

class LogBERT(nn.Module):
    """使用预训练 BERT 的日志表示模型

    直接调用 HuggingFace 上的 BERT 模型，
    截取 [CLS] 表示后接全连接层完成分类。
    """
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, x, attn_mask=None):
        out = self.bert(input_ids=x, attention_mask=attn_mask).last_hidden_state
        pooled = out[:, 0]
        return self.classifier(pooled)

class LogGD(nn.Module):
    """基于图卷积的日志表示模型

    使用两层 GraphConv 提取节点间的结构信息，
    然后平均池化得到图级表示。
    """
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        return self.classifier(h.mean(dim=0))

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
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.encoder = GraphConv(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        h = torch.relu(self.encoder(x, edge_index))
        return self.decoder(h)

class DistilBERTGraph(nn.Module):
    """DistilBERT 与 GNN 融合的多模态模型

    文本部分使用 DistilBERT 编码，图部分使用 GAT，
    将两种特征拼接后进行分类。
    """
    def __init__(self, text_model='distilbert-base-uncased', gnn_channels=64):
        super().__init__()
        self.bert = AutoModel.from_pretrained(text_model)
        self.gnn = GATConv(gnn_channels, gnn_channels)
        self.fc = nn.Linear(self.bert.config.hidden_size + gnn_channels, 2)

    def forward(self, text_ids, text_mask, x, edge_index):
        text_feat = self.bert(input_ids=text_ids, attention_mask=text_mask).last_hidden_state[:,0]
        gnn_feat = torch.relu(self.gnn(x, edge_index)).mean(dim=0)
        fused = torch.cat([text_feat, gnn_feat], dim=-1)
        return self.fc(fused)