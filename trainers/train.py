import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.topo_dataset import TopologyDataset
from datasets.alarm_dataset import AlarmDataset
from models.gnn_transformer import GNNTransformer  # 用于节点级别嵌入
from models.alarm_transformer import AlarmTransformer
from torch.nn import MultiheadAttention


def train_model():
    # 1) 加载网络拓扑数据
    topo_ds = TopologyDataset("data/topo_graph.json")
    x_dict, edge_index_dict, edge_attr_dict = topo_ds[0]
    # node_ids 是真实节点的列表，对应的索引是 1..len(node_ids)
    node_map = {nid: idx + 1 for idx, nid in enumerate(topo_ds.node_ids)}
    # 这样，node_map[...] 永远不会生成 0，0 被专门留给 PAD

    # 2) 加载告警日志数据
    alarm_ds = AlarmDataset("data/alarms.json", node_map)
    loader = DataLoader(alarm_ds, batch_size=16, shuffle=True)

    # 3) 构建 GNN 编码器
    gnn = GNNTransformer(
        in_channels=topo_ds.feature_dim,
        hidden_channels=64,
        dropout=0.3,
        num_layers=3
    )

    # 预计算一次所有节点的 embedding
    with torch.no_grad():  # 不要为它建图
        h_dict = gnn(x_dict, edge_index_dict, edge_attr_dict)
        h_core = h_dict['core']
        h_agg = h_dict['agg']
        h_access = h_dict['access']
        pad = torch.zeros(1, h_core.size(1),
                          dtype=h_core.dtype,
                          device=h_core.device)
        node_embs = torch.cat([pad, h_core, h_agg, h_access], dim=0)
    # 确保 node_embs 上没有梯度关系
    node_embs = node_embs.detach()
    # 结果形状：[1+8+20+50, H] = [79, H]

    # 4) 构建 Alarm Transformer
    # text_feat 的 shape 是 [L, feat_dim]
    feat_dim = alarm_ds[0]['text_feat'].shape[1]  # [B,L,feat_dim]
    at = AlarmTransformer(
        input_dim=feat_dim,
        emb_dim=64,
        nhead=4,
        hid_dim=128,
        max_len=alarm_ds.max_len,
        nlayers=2
    )

    #  定义跨模态注意力 & 门控网络
    cross_attn = MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.3)
    gate_net = nn.Sequential(
        nn.Linear(64 * 2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    # 5) 构建融合后的分类器头（Multi‑Task）
    fused_dim = 64   # GNN 64 + Transformer 64
    shared = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU())
    head_root = nn.Linear(128, 2)
    head_true = nn.Linear(128, 2)

    optimizer = Adam(
        list(gnn.parameters()) +
        list(at.parameters()) +
        list(cross_attn.parameters()) +
        list(gate_net.parameters()) +
        list(shared.parameters()) +
        list(head_root.parameters()) +
        list(head_true.parameters()),
        lr=1e-3
    )

    # 6) 训练循环
    for epoch in range(10):
        total_root, total_true = 0.0, 0.0
        for batch in loader:
            # 6.1 -- 从预计算的 node_embs 中抽出本批序列的 node 嵌入 --
            # batch['node_idxs']: [B, L]，先取出对应节点的嵌入 [B, L, 64]
            seq_node_embs = node_embs[batch['node_idxs']]  # [B, L, 64]
            # 对序列维度做平均，得到 [B, 64]
            node_feat = seq_node_embs.mean(dim=1)

            # 6.2 Transformer 嵌入
            # batch['text_feat'] 已经是 [B, L, feat_dim]，直接送入
            text_feat = at(batch['text_feat'])  # [B, 64]

            # 6.3 Cross-Attention 融合
            # 构造序列：2 tokens × B samples × 64 dim
            seq = torch.stack([node_feat, text_feat], dim=0)  # [2, B, 64]
            attn_out, _ = cross_attn(seq, seq, seq)  # [2, B, 64]
            # 取两 token 的平均作为跨模态输出
            attn_fused = attn_out.mean(dim=0)  # [B, 64]
            # h = torch.cat([node_feat, text_feat], dim=-1)  # [B, fused_dim]

            # 6.4 Gating Mechanism 融合
            cat = torch.cat([node_feat, text_feat], dim=-1)  # [B, 128]
            gate = gate_net(cat)  # [B, 1] in (0,1)
            fused = gate * node_feat + (1 - gate) * text_feat  # [B, 64]

            # 6.5 Multi-Task Head
            z = shared(fused)  # [B, 128]
            out_root = head_root(z)  # [B, 2]
            out_true = head_true(z)  # [B, 2]

            # 6) 损失
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

        print(f"Epoch {epoch:02d} | Loss_root={total_root / len(loader):.4f} "
              f"| Loss_true={total_true / len(loader):.4f}")


if __name__ == '__main__':
    train_model()
