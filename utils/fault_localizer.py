import os
import torch
from torch.utils.data import DataLoader
from datasets.topo_dataset import TopologyDataset
from datasets.alarm_dataset import AlarmDataset
from models.gnn_transformer import GNNTransformer
from models.alarm_transformer import AlarmTransformer
from utils.config import load_config


def localize_fault(cfg_path: str, model_dir: str, top_k: int = 1):
    """根据训练好的模型进行故障定位，返回每条序列的预测结果。

    参数:
        cfg_path: 配置文件路径
        model_dir: 模型权重所在目录，应包含 ``best_gnn.pth``、``best_at.pth``、``best_token.pth``
        top_k: 选取概率最高的前 k 个告警作为候选根因
    返回:
        List[Dict] 形如 [{'fault': True, 'node': '北京', 'device': 'dev1'}, ...]
    """
    cfg = load_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- 加载数据 -----
    topo_ds = TopologyDataset(cfg.data.topo_path)
    node_map = {nid: idx + 1 for idx, nid in enumerate(topo_ds.node_ids)}
    alarm_ds = AlarmDataset(cfg.data.alarm_path, node_map,
                            max_len=cfg.data.max_len,
                            window_milliseconds=cfg.data.window_milliseconds,
                            step_milliseconds=cfg.data.step_milliseconds)

    loader = DataLoader(alarm_ds, batch_size=1, shuffle=False)

    # ----- 加载模型 -----
    gnn = GNNTransformer(in_channels=topo_ds.feature_dim,
                         hidden_channels=cfg.gnn.hidden_channels,
                         num_layers=cfg.gnn.num_layers,
                         dropout=cfg.gnn.dropout).to(device)
    gnn.load_state_dict(torch.load(os.path.join(model_dir, 'best_gnn.pth'), map_location=device))
    at = AlarmTransformer(input_dim=alarm_ds[0]['text_feat'].shape[1],
                          emb_dim=cfg.transformer.emb_dim,
                          nhead=cfg.transformer.nhead,
                          hid_dim=cfg.transformer.hid_dim,
                          nlayers=cfg.transformer.nlayers,
                          max_len=cfg.transformer.max_len,
                          dropout=cfg.transformer.dropout).to(device)
    at.load_state_dict(torch.load(os.path.join(model_dir, 'best_at.pth'), map_location=device))
    token_head = torch.nn.Linear(cfg.transformer.emb_dim, 2).to(device)
    token_head.load_state_dict(torch.load(os.path.join(model_dir, 'best_token.pth'), map_location=device))

    gnn.eval(); at.eval(); token_head.eval()

    # 预计算节点嵌入
    with torch.no_grad():
        x_dict, edge_index_dict, edge_attr_dict = topo_ds[0]
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
        edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}
        h_dict = gnn(x_dict, edge_index_dict, edge_attr_dict)
        pad = torch.zeros(1, cfg.gnn.hidden_channels, device=device)
        node_embs = torch.cat([pad, h_dict['core'], h_dict['agg'], h_dict['access']], dim=0).detach()

    results = []
    for sample in loader:
        sample = {k: v.to(device) for k, v in sample.items()}
        # 节点特征
        seq_node_embs = node_embs[sample['node_idxs']]
        node_feat = seq_node_embs.mean(dim=1)
        # Transformer 特征及序列预测
        pooled, seq_feat = at(sample['text_feat'], return_seq=True)
        token_logits = token_head(seq_feat)
        prob = token_logits.softmax(dim=-1)[..., 1]  # [1, L]
        # 取概率最高的位置
        topk = prob.topk(top_k, dim=1)
        idx = topk.indices[0]
        is_fault = prob.max().item() > 0.5
        if is_fault:
            nid = sample['node_idxs'][0, idx[0]].item()
            did = sample['device_idxs'][0, idx[0]].item()
            node_name = alarm_ds.idx_to_node[nid] if nid < len(alarm_ds.idx_to_node) else 'UNK'
            device_id = alarm_ds.idx_to_device[did] if did < len(alarm_ds.idx_to_device) else 'UNK'
            results.append({'fault': True, 'node': node_name, 'device': device_id})
        else:
            results.append({'fault': False, 'node': None, 'device': None})
    return results


if __name__ == '__main__':
    import json
    res = localize_fault('../configs/config.yaml', '../data/processed/model_xxxx', top_k=3)
    print(json.dumps(res, ensure_ascii=False, indent=2))