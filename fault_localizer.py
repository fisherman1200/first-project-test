import os
import torch
from torch.utils.data import DataLoader
from datasets.topo_dataset import TopologyDataset
from datasets.alarm_dataset import AlarmDataset
from models.hetero_full_model import FullModel
from utils.config import load_config


def localize_fault(cfg_path: str, model_dir: str, top_k: int = 1):
    """根据训练好的模型进行故障定位，返回每条序列的预测结果。

    参数:
        cfg_path: 配置文件路径
        model_dir: 模型权重所在目录，应包含 ``best_model.pth``
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

    # ----- 加载整体模型 -----
    model = FullModel(cfg, topo_ds.feature_dim,
                      alarm_ds[0]['text_feat'].shape[1]).to(device)
    state_path = os.path.join(model_dir, 'best_model.pth')  # 模型权重文件路径
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()  # 切换到评估模式

    # 预计算并缓存所有节点嵌入
    with torch.no_grad():
        x_dict, edge_index_dict, edge_attr_dict = topo_ds[0]
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
        edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}
        model.compute_node_embs(x_dict, edge_index_dict, edge_attr_dict)

    results = []
    for sample in loader:
        # 将样本字典中的张量移动到指定设备
        sample = {k: v.to(device) for k, v in sample.items()}
        with torch.no_grad():
            # 前向推理得到序列级根因概率
            _, out_true, token_logits = model(sample)
            prob = token_logits.softmax(dim=-1)[..., 1]
        if out_true.softmax(dim=-1)[..., 1] > 0.5:
            # 取概率最高的告警位置
            topk = prob.topk(top_k, dim=1)
            idx_list = topk.indices[0]  # shape: [top_k]
            val_list = topk.values[0]  # 对应的概率值

            for i in range(top_k):
                idx = idx_list[i].item()  # 序列位置
                p = val_list[i].item()  # 概率值
                is_fault = p > 0.8

                # 取出对应的节点/设备 ID
                nid = sample['node_idxs'][0, idx].item()
                did = sample['device_idxs'][0, idx].item()

                node_name = alarm_ds.idx_to_node[nid] if nid < len(alarm_ds.idx_to_node) else 'UNK'
                device_id = alarm_ds.idx_to_device[did] if did < len(alarm_ds.idx_to_device) else 'UNK'

                results.append({
                    'fault': is_fault,
                    'node': node_name,
                    'device': device_id,
                    'prob': round(p, 4)  # 也可以把概率保存下来，方便分析
                })
    return results


if __name__ == '__main__':
    import json
    res = localize_fault('configs/config.yaml', 'data/processed/model/20250820_035420', top_k=3)
    print(json.dumps(res, ensure_ascii=False, indent=2))
