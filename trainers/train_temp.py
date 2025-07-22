import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.topo_dataset import TopologyDataset
from datasets.alarm_dataset import AlarmDataset
from models.gnn_transformer import GNNTransformer  # 用于节点级别嵌入
from models.alarm_transformer import AlarmTransformer

def train_model():
    # 1) 加载网络拓扑数据
    topo_ds = TopologyDataset(json_path="data/topo_graph.json")
    topo_data = topo_ds.data
    # node_ids 是真实节点的列表，对应的索引是 1..len(node_ids)
    node_map = {nid: idx + 1 for idx, nid in enumerate(topo_ds.node_ids)}
    # 这样，node_map[...] 永远不会生成 0，0 被专门留给 PAD

    # 2) 加载告警日志数据
    alarm_ds = AlarmDataset("data/alarms.json", node_map)
    loader = DataLoader(alarm_ds, batch_size=16, shuffle=True)

    # 3) 构建 GNN 编码器
    gnn = GNNTransformer(
        in_channels=topo_ds.num_features,
        edge_dim=None,  # 这一项不再需要
        hidden_channels=64,
        heads=4
    )

    # 4) 构建 Alarm Transformer
    # text_feat 的 shape 是 [L, feat_dim]，我们需要 feat_dim
    feat_dim = alarm_ds[0]['text_feat'].shape[1]
    at = AlarmTransformer(
        input_dim=feat_dim,
        emb_dim=64,
        nhead=4,
        hid_dim=128,
        nlayers=2
    )

    # 5) 构建融合后的分类器头（Multi‑Task）
    fused_dim = 64 + 64  # GNN 64 + Transformer 64
    shared = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU())
    head_root = nn.Linear(128, 2)
    head_true = nn.Linear(128, 2)

    optimizer = Adam(
        list(gnn.parameters()) +
        list(at.parameters()) +
        list(shared.parameters()) +
        list(head_root.parameters()) +
        list(head_true.parameters()),
        lr=1e-3
    )

    # 6) 训练循环
    for epoch in range(10):
        total_root, total_true = 0.0, 0.0
        for batch in loader:
            # 6.1 先算出所有节点的嵌入
            node_embs = gnn(
                topo_data.x,
                topo_data.edge_index)  # [N, 64]
            # 取出本 batch 样本对应节点的嵌入
            # batch['node_idxs']: [B, L]，先取出对应节点的嵌入 [B, L, 64]
            seq_node_embs = node_embs[batch['node_idxs']]  # [B, L, 64]
            # 对序列维度做平均，得到 [B, 64]
            node_feat = seq_node_embs.mean(dim=1)

            # 6.2 Transformer 嵌入
            # batch['text_feat'] 已经是 [B, L, feat_dim]，直接送入
            text_feat = at(batch['text_feat'])  # [B, 64]

            # 6.3 拼接
            h = torch.cat([node_feat, text_feat], dim=-1)  # [B, fused_dim]

            # 6.4 共享 MLP + 两头输出
            z = shared(h)                                 # [B, 128]
            out_root = head_root(z)                       # [B, 2]
            out_true = head_true(z)                       # [B, 2]

            # 6.5 Multi‑Task 损失
            is_root_seq = batch['is_root']  # [B, L]
            is_true_seq_fault = batch['is_true_fault']  # [B, L]
            # 取第一个元素
            root_label = is_root_seq[:, 0]  # [B]
            true_label = is_true_seq_fault[:, 0]  # [B]
            loss_root = F.cross_entropy(out_root, root_label)
            mask = (root_label == 1)
            if mask.any():
                loss_true = F.cross_entropy(out_true[mask], true_label[mask])
            else:
                loss_true = torch.tensor(0.0, device=h.device)
            loss = loss_root + 2.0 * loss_true

            total_root += loss_root.item()
            total_true += loss_true.item()

            # 6.6 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch:02d} | Loss_root={total_root/len(loader):.4f} "
              f"| Loss_true={total_true/len(loader):.4f}")

if __name__ == '__main__':
    train_model()
