import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from datasets.topo_dataset import TopologyDataset
from datasets.alarm_dataset import AlarmDataset
from models.gnn_transformer import GNNTransformer  # 用于节点级别嵌入
from models.alarm_transformer import AlarmTransformer
from torch.nn import MultiheadAttention
from tqdm import tqdm
import json
from utils.metrics_utils import MetricsLogger
from utils.visualize_embeddings import extract_embeddings, plot_tsne
from utils.model_utils import ModelSaver
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler


def train_model(cfg):
    # 启动gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    # 1) 加载网络拓扑数据
    topo_ds = TopologyDataset(cfg.data.topo_path)
    x_dict, edge_index_dict, edge_attr_dict = topo_ds[0]
    # node_ids 是真实节点的列表，对应的索引是 1..len(node_ids)
    node_map = {nid: idx + 1 for idx, nid in enumerate(topo_ds.node_ids)}
    # 这样，node_map[...] 永远不会生成 0，0 被专门留给 PAD

    # 2) 加载告警日志数据
    full_alarm_ds = AlarmDataset(cfg.data.alarm_path, node_map, max_len=cfg.data.max_len,
                                 window_milliseconds=cfg.data.window_milliseconds, step_milliseconds=cfg.data.step_milliseconds)
    # 划分训练集、验证集
    train_ds, val_ds, test_ds = random_split(full_alarm_ds, [cfg.data.num_train, cfg.data.num_val, cfg.data.num_test],
                                             generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)

    # 3) 构建 GNN 编码器
    gnn = GNNTransformer(
        in_channels=topo_ds.feature_dim,
        hidden_channels=cfg.gnn.hidden_channels,
        dropout=cfg.gnn.dropout,
        num_layers=cfg.gnn.num_layers
    ).to(device)

    # 把 x_dict 中所有张量移到 device
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

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
    feat_dim = full_alarm_ds[0]['text_feat'].shape[1]  # [B,L,feat_dim]
    at = AlarmTransformer(
        input_dim=full_alarm_ds[0]['text_feat'].shape[1],
        emb_dim=cfg.transformer.emb_dim,
        nhead=cfg.transformer.nhead,
        hid_dim=cfg.transformer.hid_dim,
        nlayers=cfg.transformer.nlayers,
        max_len=cfg.transformer.max_len,
        dropout=cfg.transformer.dropout
    ).to(device)

    #  定义跨模态注意力 & 门控网络
    cross_attn = MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.3)
    gate_net = nn.Sequential(
        nn.Linear(64 * 2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    # 5) 构建融合后的分类器头（Multi‑Task）
    fused_dim = 64  # GNN 64 + Transformer 64
    shared = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU())
    head_root = nn.Linear(128, 2)
    head_true = nn.Linear(128, 2)

    # 全部迁移到 GPU
    for m in [cross_attn, gate_net, shared, head_root, head_true]:
        m.to(device)

    optimizer = Adam(
        list(gnn.parameters()) +
        list(at.parameters()) +
        list(cross_attn.parameters()) +
        list(gate_net.parameters()) +
        list(shared.parameters()) +
        list(head_root.parameters()) +
        list(head_true.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    # 选择一个 scheduler，学习率调度：
    scheduler = StepLR(optimizer, step_size=cfg.training.lr_step_size, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # AMP
    scaler = GradScaler()

    # ————— 早停变量 —————
    best_val = float('inf')
    no_improve_times = 0

    # 初始化 Logger，指定要记录的关键指标名
    logger = MetricsLogger(keys=['train_loss', 'train_root_loss', 'train_true_loss',
                                 'val_loss', 'val_root_loss', 'val_true_loss'])  # 日后可加 'acc', 'lr' …

    # utils-> model_utils: 训练结束后保存神经网络模型数据
    saver = ModelSaver(base_dir='data/processed')

    # 6) 训练循环
    for epoch in range(cfg.training.epochs):
        total_train_loss, total_train_root_loss, total_train_true_loss = 0.0, 0.0, 0.0
        print(f"------------------Epoch :{epoch:02d},Start------------------")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():  # 自动混合精度上下文
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
                # fused = gate * node_feat + (1 - gate) * text_feat  # [B, 64]
                # —— 或者用 attn_fused 参与 gating：
                fused = gate * attn_fused + (1 - gate) * text_feat

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
                    loss_true = torch.tensor(0.0, device=fused.device)
                loss = loss_root + 2.0 * loss_true

            total_train_loss += loss.item()
            total_train_root_loss += loss_root.item()
            total_train_true_loss += loss_true.item()

            # 6.6 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_root = total_train_root_loss / len(train_loader)
        avg_train_true = total_train_true_loss / len(train_loader)
        print(
            f"Train_loss={avg_train_loss:.4f} | Train_loss_root={avg_train_root:.4f} | Train_loss_true={avg_train_true:.4f}")

        # ---- Validation ----
        gnn.eval()
        at.eval()
        shared.eval()
        total_val_loss, total_val_root_loss, total_val_true_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}"):
                batch = {k: v.to(device) for k, v in batch.items()}
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
                # fused = gate * node_feat + (1 - gate) * text_feat  # [B, 64]
                # —— 或者用 attn_fused 参与 gating：
                fused = gate * attn_fused + (1 - gate) * text_feat

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
                    loss_true = torch.tensor(0.0, device=fused.device)
                val_loss = loss_root + 2.0 * loss_true

            total_val_loss += val_loss.item()
            total_val_root_loss += loss_root.item()
            total_val_true_loss += loss_true.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_root = total_val_root_loss / len(val_loader)
        avg_val_true = total_val_true_loss / len(val_loader)
        print(f"Val_loss={avg_val_loss:.4f} | Val_loss_root={avg_val_root:.4f} | Val_loss_true={avg_val_true:.4f}")
        print(f"------------------Epoch :{epoch:02d},End------------------")

        #  添加到 logger
        avg_metrics = {'train_loss': avg_train_loss,
                       'train_root_loss': avg_train_root,
                       'train_true_loss': avg_train_true,
                       'val_loss': avg_val_loss,
                       'val_root_loss': avg_val_root,
                       'val_true_loss': avg_val_true}
        logger.add(epoch, avg_metrics)

        # —— Scheduler & Early Stopping ——
        scheduler.step()
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            no_improve_times = 0
            saved = saver.save_best(gnn=gnn, at=at)
        else:
            no_improve_times += 1
            if no_improve_times >= cfg.training.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # —— 训练 & 验证 结束后，加载最佳模型权重，再对 test_loader 评估一次 ——
    print("Loading best model weights and evaluating on TEST set…")
    best_gnn = GNNTransformer(
        in_channels=topo_ds.feature_dim,
        hidden_channels=cfg.gnn.hidden_channels,
        dropout=cfg.gnn.dropout,
        num_layers=cfg.gnn.num_layers
    )
    best_gnn.load_state_dict(torch.load(saved['gnn']))
    best_at = AlarmTransformer(
        input_dim=full_alarm_ds[0]['text_feat'].shape[1],
        emb_dim=cfg.transformer.emb_dim,
        nhead=cfg.transformer.nhead,
        hid_dim=cfg.transformer.hid_dim,
        nlayers=cfg.transformer.nlayers,
        max_len=cfg.transformer.max_len,
        dropout=cfg.transformer.dropout
    )
    best_at.load_state_dict(torch.load(saved['at']))
    best_gnn.eval()
    best_at.eval()

    # 测试评估
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Number of test batches: {len(test_loader)}")

    total_test_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # 6.1 -- 从预计算的 node_embs 中抽出本批序列的 node 嵌入 --
            # batch['node_idxs']: [B, L]，先取出对应节点的嵌入 [B, L, 64]
            seq_node_embs = node_embs[batch['node_idxs']]  # [B, L, 64]
            batch_size = seq_node_embs.size(0)  # 当前这个 batch 的样本数 B
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
            # fused = gate * node_feat + (1 - gate) * text_feat  # [B, 64]
            # —— 或者用 attn_fused 参与 gating：
            fused = gate * attn_fused + (1 - gate) * text_feat

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
                loss_true = torch.tensor(0.0, device=fused.device)
            test_loss = loss_root + 2.0 * loss_true
            total_test_loss += test_loss.item() * batch_size
            total_samples += batch_size

    print(f"Final TEST Loss: {total_test_loss / total_samples:.4f}")
    # 也可以计算 accuracy、precision/recall 等指标

    # —— 可视化 & 其他后处理 ——
    # utils-> plot_metrics: 生成metrics数据json  TODO:所有metrics都画在一张图上，要根据需求分开
    logger.save()

    # utils-> visualize_embeddings 绘制聚类效果图
    # 提取 embeddings 和标签
    embs, is_root, is_true = extract_embeddings(cfg, gnn, at, topo_ds, full_alarm_ds)
    # 根源 vs 衍生
    plot_tsne(embs, is_root, 't-SNE: Root vs Derived', "tsne_root_vs_derived", names=('Derived', 'Root'))
    # 真故障根源 vs 非真故障根源
    mask_root = is_root == 1
    plot_tsne(embs[mask_root], is_true[mask_root], 't-SNE: True Fault among Root',
              "tsne_true_among_root", names=('Non-Fault', 'True Fault'))
