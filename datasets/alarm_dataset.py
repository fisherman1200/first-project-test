import json
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

import os

# 1) 定位到 datasets 目录
base_dir = os.path.dirname(os.path.abspath(__file__))
# 2) 顶层项目根目录
proj_root = os.path.abspath(os.path.join(base_dir, os.pardir))
# 3) 构造 topo_graph.json 和 alarms.json 的绝对路径
topo_json = os.path.join(proj_root, 'data', 'topo_graph.json')
alarm_json = os.path.join(proj_root, 'data', 'alarms.json')


class AlarmDataset(Dataset):
    """
    返回：
    node_idxs:  类型：LongTensor，形状 [L]，这里 L = max_len
                含义：序列中每条告警对应的网络拓扑节点的索引。比如窗口里第0个告警发生在 “上海市”，
                在 TopologyDataset 里我们把 “上海市” 映射成节点10，那么 node_idxs[0] = 10；
                如果序列被填充（padding），多余的位置就重复最后一个合法节点索引或填 0。
    text_feat:  类型：FloatTensor，形状 [L, D]，其中 D 是每条告警的特征维度（如 alarm_type的one‑hot 维度 + severity + timestamp）。
                含义：将告警的文本/数值字段编码成向量，
                通常包括：
                    告警类型（alarm_type）的 One‑Hot 编码
                    严重度（severity）映射成数值（如 Critical=2, Major=1, Minor=0）
                    时间戳（timestamp）归一化后的实数
    is_root:
    is_true_fault:
    """

    def __init__(self, json_path, node_id_map, max_len=16,
                 window_minutes=10, step_minutes=5):
        # 如果外面给的 map 为空或坐标对不上，就自己从 topo_graph.json 里读
        if node_id_map is None or not node_id_map:
            with open(topo_json, 'r', encoding='utf-8-sig') as f:
                topo = json.load(f)
            # 建立一个干净的 map：ID -> idx+1
            node_id_map = {n['id'].strip(): i + 1
                           for i, n in enumerate(topo['nodes'])}
        self.node_id_map = node_id_map
        self.max_len = max_len

        # Load raw alarms (GBK encoding for Chinese)
        with open(json_path, 'r', encoding='gbk') as f:
            raw = json.load(f)
        # Parse timestamps and sort
        alarms = []
        for item in raw:
            ts = datetime.fromisoformat(item['timestamp'])
            item['_ts'] = ts
            alarms.append(item)
        alarms.sort(key=lambda x: x['_ts'])

        # Sliding window
        ws = timedelta(minutes=window_minutes)
        st = timedelta(minutes=step_minutes)
        start = alarms[0]['_ts']
        end = alarms[-1]['_ts']
        sequences = []
        sid = 0
        while start <= end:
            window = [a for a in alarms if start <= a['_ts'] < start + ws]
            if window:
                sequences.append(window)
                sid += 1
            start += st

        '''更改：从one-hot到RoBERTa
        # Build global type encoder
        types = [[a['alarm_type']] for seq in sequences for a in seq]
        self.enc_type = OneHotEncoder(sparse_output=False).fit(np.array(types))
        # 在 fit 完后，确定每条告警的向量维度：
        type_dim = len(self.enc_type.categories_[0])
        self.feat_dim = type_dim + 2  # alarm_type one-hot + severity + timestamp
        '''
        # ---------------------------更改：从one-hot到RoBERTa---------------------------
        # ——— 加载预训练语言模型 ———
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.text_encoder.eval()
        # 文本特征维度（roberta-base hidden size = 768）
        self.feat_dim = self.text_encoder.config.hidden_size
        # 如果有 GPU，可将模型移动到 GPU
        if torch.cuda.is_available():
            self.text_encoder.to('cuda')
        # —— 新增：node_name 和 device_id 的 LabelEncoder + Embedding ——
        # 先从 raw alarms 里收集所有可能的值
        all_node_names = sorted({a['node_name'].strip() for a in raw})
        all_device_ids = sorted({a['device_id'] for a in raw})
        self.le_node = LabelEncoder().fit(all_node_names)
        self.le_dev = LabelEncoder().fit(all_device_ids)
        # 定义 embedding 层
        self.node_emb = nn.Embedding(len(self.le_node.classes_), 32)
        self.dev_emb = nn.Embedding(len(self.le_dev.classes_), 32)
        # 如果有 GPU，就一起搬过去
        if torch.cuda.is_available():
            self.node_emb.to('cuda')
            self.dev_emb.to('cuda')
        # ---------------------------更改：从one-hot到RoBERTa---------------------------

        # Build samples: pad/truncate each sequence
        self.samples = []
        for seq in sequences:
            node_idxs, feats, root_labels, true_labels = [], [], [], []

            for a in seq:
                # —— 1) 先做一下清洗 ——
                raw_name = a.get('node_name', '')
                # 去掉左右空白、全角空格，以及常见 BOM
                name = raw_name.strip().strip('\ufeff\u3000')
                nid = self.node_id_map.get(name)
                if nid is None:
                    # ⚠️ 仍然打印一次，方便后续确认
                    #  print("⚠️ 找不到节点映射（已 PAD）:", repr(raw_name), "-> clean:", repr(name))
                    nid = 0  # PAD 索引
                node_idxs.append(nid)

                '''更改：从one-hot到RoBERTa
                # 构造 feature 向量
                type_vec = self.enc_type.transform([[a['alarm_type']]])[0]
                sev_map = {'Critical': 2, 'Major': 1, 'Minor': 0}
                sev = sev_map.get(a['severity'], 0)
                ts_norm = a['_ts'].timestamp() / 1e9
                feats.append(np.concatenate([type_vec, [sev, ts_norm]]))
                '''
                # ---------------------------更改：从one-hot到RoBERTa---------------------------
                # —— 新：用 RoBERTa 提取深度语义 ——
                # —— 1) 文本 Embedding ——
                alarm_text = f"{a['alarm_type']}。{a.get('comment', '')}"
                inputs = self.tokenizer(
                    alarm_text,
                    truncation=True,
                    padding='max_length',
                    max_length=64,
                    return_tensors='pt'
                )
                # 移动到 GPU（如可用）
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.text_encoder(**inputs)
                # 取 [CLS] token 表示，转回 CPU 并转 numpy
                cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
                text_emb = cls_emb  # torch tensor [768,]

                # —— 2) severity 和 timestamp ——
                sev_map = {'Critical': 2, 'Major': 1, 'Minor': 0}
                sev = sev_map.get(a['severity'], 0)
                ts_norm = a['_ts'].timestamp() / 1e9

                # —— 3) node_name & device_id Embedding ——
                nidx = self.le_node.transform([a['node_name'].strip()])[0]
                didx = self.le_dev.transform([a['device_id']])[0]
                node_v = self.node_emb(torch.tensor(nidx, device=text_emb.device))
                dev_v = self.dev_emb(torch.tensor(didx, device=text_emb.device))

                # —— 4) concat 所有特征 ——
                # [768] ∥ [1] ∥ [1] ∥ [32] ∥ [32] → [834]
                feat = torch.cat([
                        text_emb,
                        torch.tensor([sev, ts_norm], device=text_emb.device),
                        node_v,
                        dev_v
                ], dim=0)
                feats.append(feat.detach().cpu().numpy())
                # ---------------------------更改：从one-hot到RoBERTa---------------------------

                # 构造标签
                c = a.get('comment', '').lower()
                port_status = a.get('port', {}).get('status', '').lower()
                if 'rootcause' in c:
                    root_labels.append(1)
                    true_labels.append(0 if port_status == 'maintenance' else 1)
                else:
                    root_labels.append(0)
                    true_labels.append(0)

            # 准备固定的 padding 向量
            self.feat_dim = 834
            pad_feat = np.zeros(self.feat_dim, dtype=float)

            # pad/truncate 到 max_len
            L = len(feats)
            pad_n = max_len - L if L < max_len else 0
            if L < max_len:
                feats.extend([pad_feat] * pad_n)
                node_idxs.extend([0] * pad_n)
                root_labels.extend([0] * pad_n)
                true_labels.extend([0] * pad_n)
            elif L > max_len:
                feats = feats[:max_len]
                node_idxs = node_idxs[:max_len]
                root_labels = root_labels[:max_len]
                true_labels = true_labels[:max_len]

            # 转成 tensor 并保存
            self.samples.append({
                'node_idxs': torch.tensor(node_idxs, dtype=torch.long),
                'text_feat': torch.tensor(feats, dtype=torch.float),
                'is_root': torch.tensor(root_labels, dtype=torch.long),
                'is_true_fault': torch.tensor(true_labels, dtype=torch.long)
            })

        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    with open(topo_json, 'r', encoding='utf-8') as f:
        topo = json.load(f)
    # 建立一个干净的 map：ID -> idx+1
    node_id_map = {n['id'].strip(): i + 1
                   for i, n in enumerate(topo['nodes'])}

    # 构建并导出 AlarmDataset
    ds = AlarmDataset(alarm_json, node_id_map, max_len=16, window_minutes=10, step_minutes=5)

    out = []
    for sample in ds:
        out.append({
            'node_idxs': sample['node_idxs'].tolist(),
            'text_feat': sample['text_feat'].tolist(),
            'is_root': sample['is_root'].tolist(),
            'is_true_fault': sample['is_true_fault'].tolist()
        })

    # 导出到 data/processed_alarm_sequences.json
    output_path = os.path.join(proj_root, 'data', 'processed_alarm_sequences.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        import json

        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"已导出 {len(out)} 条告警序列到 {output_path}")
