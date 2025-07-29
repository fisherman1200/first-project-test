import os
import torch
from torch.optim import Adam
from torch_geometric.utils import negative_sampling
from datasets.topo_dataset import TopologyDataset
from models.gnn_transformer import GNNTransformer


def pretrain_gnn(cfg, epochs: int = 50):
    """简单的无监督边预测任务, 预训练GNN参数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = TopologyDataset(cfg.data.topo_path)
    x_dict, edge_index_dict, edge_attr_dict = ds[0]
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

    model = GNNTransformer(
        in_channels=ds.feature_dim,
        hidden_channels=cfg.gnn.hidden_channels,
        num_layers=cfg.gnn.num_layers,
        dropout=cfg.gnn.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        h_dict = model(x_dict, edge_index_dict, edge_attr_dict)
        total_loss = 0.0
        for rel, eidx in edge_index_dict.items():
            src_type, _, tgt_type = rel
            pos_src = h_dict[src_type][eidx[0]]
            pos_tgt = h_dict[tgt_type][eidx[1]]
            pos_score = (pos_src * pos_tgt).sum(dim=1)
            neg_eidx = negative_sampling(
                eidx,
                num_nodes=(x_dict[src_type].size(0), x_dict[tgt_type].size(0)),
                num_neg_samples=eidx.size(1),
                method='sparse',
            )
            neg_src = h_dict[src_type][neg_eidx[0]]
            neg_tgt = h_dict[tgt_type][neg_eidx[1]]
            neg_score = (neg_src * neg_tgt).sum(dim=1)
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ])
            total_loss = total_loss + criterion(scores, labels)
        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")

    os.makedirs(os.path.dirname(cfg.gnn.pretrained_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.gnn.pretrained_path)
    print(f"预训练完成，权重已保存到 {cfg.gnn.pretrained_path}")