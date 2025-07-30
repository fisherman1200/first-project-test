#!/usr/bin/env python3
"""
visualize_embeddings.py

使用示例：
  python visualize_embeddings.py \
    --config config.yaml \
    --ckpt_gnn path/to/gnn_model.pth \
    --ckpt_at path/to/transformer_model.pth \
    --output_dir data/processed/embeddings_vis

功能：
1. 加载配置、模型权重。
2. 对整个图进行 GNN 推理，预计算节点嵌入。
3. 对每条告警序列，平均池化对应节点嵌入，并拼接 Transformer 文本嵌入。
4. 使用 t-SNE 将高维融合嵌入降到 2D，并根据标签（是否根源、是否真实故障）画散点图，
   图表风格与论文及 PPT 中的 loss 曲线一致。
"""
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.config import load_config
from datasets.topo_dataset import TopologyDataset
from datasets.alarm_dataset import AlarmDataset
from models.gnn_transformer import GNNTransformer
from models.alarm_transformer import AlarmTransformer
from utils.path_utils import get_output_dir

# Matplotlib 全局配置，使用 LaTeX 字体，统一论文级规范
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (6, 6),
    "lines.linewidth": 1.5,
    "lines.markersize": 8,
    "savefig.dpi": 300,
})


def extract_embeddings(cfg, gnn, at, topo_ds, alarm_ds):
    device = next(gnn.parameters()).device  # 自动获取 GNN 模型所在设备
    # 1) 预计算节点嵌入
    gnn.eval()
    with torch.no_grad():
        h_dict = gnn(
            {k: v.to(device) for k, v in topo_ds.x_dict.items()},
            {k: v.to(device) for k, v in topo_ds.edge_index_dict.items()},
            {k: v.to(device) for k, v in topo_ds.edge_attr_dict.items()}
        )
    # 合并并加 PAD
    h_core = h_dict['core']; h_agg = h_dict['agg']; h_access = h_dict['access']
    pad = torch.zeros(1, h_core.size(1), device=h_core.device)
    node_embs = torch.cat([pad, h_core, h_agg, h_access], dim=0).detach()

    # 2) 提取序列嵌入与标签
    embs, is_root, is_true = [], [], []
    at.eval()
    for sample in alarm_ds:
        idx = sample['node_idxs'].to(node_embs.device)
        # 节点平均池化
        pool_node = node_embs[idx].mean(dim=0, keepdim=True)
        # 文本嵌入
        with torch.no_grad():
            txt_emb = at(sample['text_feat'].unsqueeze(0).to(pool_node.device))
        # 拼接并 detach
        emb_tensor = torch.cat([pool_node, txt_emb], dim=1).squeeze(0)
        emb = emb_tensor.detach().cpu().numpy()
        embs.append(emb)
        is_root.append(int(sample['is_root'].max().item()))
        is_true.append(int(sample['is_true_fault'].max().item()))
    return np.stack(embs), np.array(is_root), np.array(is_true)


def plot_tsne(embs: np.ndarray, labels: np.ndarray, title: str, file_name: str, names=('A','B')):
    """
    在一张图中，用不同形状和颜色对比两类点的 t-SNE 聚类效果。
    labels: 0/1 的二分类标签
    labels==0 对应 names[0], labels==1 对应 names[1]
    """
    # 分布检查
    u, c = np.unique(labels, return_counts=True)
    dist = dict(zip(u, c))
    print(f"{title} distribution {{0:'{names[0]}',1:'{names[1]}'}} => {dist}")
    # 自动设置 perplexity
    n_samples = embs.shape[0]
    perp = min(30, max(1, n_samples // 3))
    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=perp)
    vis = tsne.fit_transform(embs)

    # 绘制
    plt.figure()
    for val, mark, clr, nm in [(0, 'o', 'C0', names[0]), (1, '^', 'C1', names[1])]:
        idx = labels == val
        plt.scatter(vis[idx, 0], vis[idx, 1], marker=mark, c=clr, label=nm, s=40)
    plt.xlabel(r'$\mathrm{t	ext{-}SNE\ Dimension\ 1}$')
    plt.ylabel(r'$\mathrm{t	ext{-}SNE\ Dimension\ 2}$')
    plt.title(title)
    plt.legend(frameon=False, loc='upper right')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    # 生成带统一时间戳的输出目录
    default_dir = get_output_dir('data', 'processed', 'clustering')
    filename = f'{file_name}.pdf'
    output_path = os.path.join(default_dir, filename)
    os.makedirs(default_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved t-SNE plot '{title}' to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings via t-SNE with paper-style plots")
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--ckpt_gnn', type=str, required=True)
    parser.add_argument('--ckpt_at', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, default='data/processed/embeddings_vis')
    args = parser.parse_args()

    cfg = load_config(args.config)
    topo_ds = TopologyDataset(cfg.data.topo_path)
    alarm_ds = AlarmDataset(cfg.data.alarm_path, topo_ds.node_map, max_len=cfg.data.max_len)

    gnn = GNNTransformer(
        in_channels=topo_ds.feature_dim,
        hidden_channels=cfg.gnn.hidden_channels,
        dropout=cfg.gnn.dropout,
        num_layers=cfg.gnn.num_layers
    )
    at = AlarmTransformer(
        input_dim=alarm_ds[0]['text_feat'].shape[1],
        emb_dim=cfg.transformer.emb_dim,
        nhead=cfg.transformer.nhead,
        hid_dim=cfg.transformer.hid_dim,
        nlayers=cfg.transformer.nlayers,
        max_len=cfg.transformer.max_len,
        dropout=cfg.transformer.dropout
    )
    gnn.load_state_dict(torch.load(args.ckpt_gnn))
    at.load_state_dict(torch.load(args.ckpt_at))
    gnn.cpu(); at.cpu()

    embs, root_labels, true_labels = extract_embeddings(cfg, gnn, at, topo_ds, alarm_ds)

    plot_tsne(embs, root_labels, 't-SNE: Root vs Derived',
              os.path.join(args.output_dir, 'tsne_root.pdf'),
              names=('Derived', 'Root'))
    mask_root = root_labels == 1
    plot_tsne(embs[mask_root], true_labels[mask_root],
              't-SNE: True Fault among Root',
              os.path.join(args.output_dir, 'tsne_true.pdf'),
              names=('Non-Fault', 'True Fault'))

if __name__ == '__main__':
    main()
