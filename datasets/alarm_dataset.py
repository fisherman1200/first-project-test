import bisect
import json
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
# from sklearn.preprocessing import OneHotEncoder
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import os
import time


class AlarmDataset(Dataset):
    """
        AlarmDataset: 从告警日志中构建时序事件序列，生成节点索引、文本特征和标签。

        输入参数:
             json_path (str): 原始告警日志 JSON 路径，GBK 编码
             node_id_map (Dict[str,int]): 拓扑节点 ID -> 索引（1 开始，0 保留为 PAD）,通常由topo_dataset得到
             max_len (int): 每个时间窗口序列的最大长度 L
             window_milliseconds (int): 粗聚集时间窗口大小（毫秒）
             step_milliseconds (int): 精细滑动步长（毫秒）
             preproc_path (str): 预处理缓存文件路径，若存在则直接加载

        输出样本 (dict) 字段:
            'node_idxs': LongTensor([L]) 序列内每条告警对应的节点索引
            'device_idxs': LongTensor([L]) 每条告警对应的设备索引
            'text_feat': FloatTensor([L, D]) 每条告警的文本+数值+嵌入特征
            'is_root': LongTensor([L]) 每条告警是否为根源 (0/1)
            'is_true_fault': LongTensor([L]) 每条根源告警是否为真实故障 (0/1)

        使用方法示例:
             ds = AlarmDataset("data/alarms.json", node_id_map, max_len=16, window_milliseconds=5000, step_milliseconds=500)
             sample = ds[0]
             node_idxs, text_feat, is_root, is_true = (
                        sample['node_idxs'], sample['text_feat'],
                        sample['is_root'], sample['is_true_fault'])
    """
    def __init__(self, json_path, node_id_map, max_len=16,
                 window_milliseconds=5000, step_milliseconds=500,
                 preproc_path='data/processed/processed_alarm_sequences_v1.pt'):
        if os.path.exists(preproc_path):
            # 直接加载预处理好的序列和映射信息
            cached = torch.load(preproc_path)
            self.samples = cached['samples']
            self.idx_to_node = cached.get('idx_to_node', [])
            self.idx_to_device = cached.get('idx_to_device', [])
            self.le_node = LabelEncoder().fit(self.idx_to_node)
            self.le_dev = LabelEncoder().fit(self.idx_to_device)
            print(f"在 {preproc_path} 发现预处理过的文件，读取 {len(self.samples)} 条数据")
            # 加载完成后统计数据集中告警分布
            self.report_stats()
        else:
            print(f"没有在 {preproc_path} 找到预处理过的数据，开始预处理数据")
            start_time = time.time()  # 记录预处理开始时间
            # 如果外面给的 map 为空或坐标对不上，就自己从 topo_graph.json 里读
            if node_id_map is None or not node_id_map:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
                    topo = json.load(f)
                # 建立一个干净的 map：ID -> idx+1
                node_id_map = {n['id'].strip(): i + 1
                               for i, n in enumerate(topo['nodes'])}
            self.node_id_map = node_id_map
            self.max_len = max_len
            # 读取原始日志
            with open(json_path, 'r', encoding='gbk') as f:
                raw_data = json.load(f)
            # 解析并排序
            print(f"读取了 {len(raw_data)} 条告警日志")
            alarms = []
            for item in raw_data:
                ts = datetime.fromisoformat(item['timestamp'])
                item['_ts'] = ts
                alarms.append(item)
            alarms.sort(key=lambda x: x['_ts'])

            # 滑动窗口聚类
            # 预先提取所有时间戳
            timestamps = [a['_ts'] for a in alarms]  # 有序

            # 粗聚集：基于 window_milliseconds
            clusters = []
            for i, t0 in enumerate(timestamps):
                j = bisect.bisect_right(timestamps, t0 + timedelta(milliseconds=window_milliseconds))
                if j > i:
                    clusters.append(alarms[i:j])

            # 细滑窗：基于 step_milliseconds
            sequences = []
            for cluster in clusters:
                # cluster 已是一个事件簇的所有告警
                ts_cluster = [a['_ts'] for a in cluster]
                for start_idx, ts_start in enumerate(ts_cluster):
                    end_idx = bisect.bisect_right(ts_cluster, ts_start + timedelta(milliseconds=step_milliseconds))
                    if end_idx > start_idx:
                        sequences.append(cluster[start_idx:end_idx])

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
            os.environ["TRANSFORMERS_OFFLINE"] = "1" # 云服务无法联网，加载离线文件
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                                              local_files_only=True, trust_remote_code=False)
            self.text_encoder = RobertaModel.from_pretrained('roberta-base',
                                                             local_files_only=True, trust_remote_code=False)
            self.text_encoder.eval()
            # 文本特征维度（roberta-base hidden size = 768）
            self.feat_dim = self.text_encoder.config.hidden_size
            # 如果有 GPU，可将模型移动到 GPU
            if torch.cuda.is_available():
                self.text_encoder.to('cuda')
            # 特征维度 = RoBERTa(768) + severity&ts(2) + node_name_emb(32) + dev_id_emb(32)
            self.feat_dim = 768 + 2 + 32 + 32
            # 节点与设备唯一值嵌入
            all_node_names = sorted({a['node_name'].strip() for a in raw_data})
            all_device_ids = sorted({a['device_id'] for a in raw_data})
            self.le_node = LabelEncoder().fit(all_node_names)
            self.le_dev = LabelEncoder().fit(all_device_ids)
            self.idx_to_node = self.le_node.classes_.tolist()
            self.idx_to_device = self.le_dev.classes_.tolist()
            # 定义 embedding 层
            self.node_emb = nn.Embedding(len(self.le_node.classes_), 32)
            self.dev_emb = nn.Embedding(len(self.le_dev.classes_), 32)
            # 如果有 GPU，就一起搬过去
            if torch.cuda.is_available():
                self.node_emb.to('cuda')
                self.dev_emb.to('cuda')
            # ---------------------------更改：从one-hot到RoBERTa---------------------------
            # 构建样本列表
            self.samples = []
            # 保存设备索引，便于后续故障定位
            for seq in sequences:
                node_idxs, device_idxs = [], []
                features, root_labels, true_labels = [], [], []
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
                    # —— 文本 Embedding ——
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

                    # ——  severity 和 timestamp ——
                    sev_map = {'Critical': 2, 'Major': 1, 'Minor': 0}
                    sev = sev_map.get(a['severity'], 0)
                    ts_norm = a['_ts'].timestamp() / 1e9

                    # ——  node_name & device_id Embedding ——
                    nidx = self.le_node.transform([a['node_name'].strip()])[0]
                    didx = self.le_dev.transform([a['device_id']])[0]
                    node_v = self.node_emb(torch.tensor(nidx, device=text_emb.device))
                    dev_v = self.dev_emb(torch.tensor(didx, device=text_emb.device))
                    device_idxs.append(didx)

                    # ——  concat 所有特征 ——
                    # [768] ∥ [1] ∥ [1] ∥ [32] ∥ [32] → [834]
                    feat = torch.cat([
                        text_emb,
                        torch.tensor([sev, ts_norm], device=text_emb.device),
                        node_v,
                        dev_v
                    ], dim=0)
                    features.append(feat.detach().cpu().numpy())
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
                seq_len = len(features)
                pad_count = max_len - seq_len if seq_len < max_len else 0
                if seq_len < max_len:
                    features.extend([pad_feat] * pad_count)
                    node_idxs.extend([0] * pad_count)
                    device_idxs.extend([0] * pad_count)
                    root_labels.extend([0] * pad_count)
                    true_labels.extend([0] * pad_count)
                elif seq_len > max_len:
                    features = features[:max_len]
                    node_idxs = node_idxs[:max_len]
                    device_idxs = device_idxs[:max_len]
                    root_labels = root_labels[:max_len]
                    true_labels = true_labels[:max_len]

                # 新：先堆成大 numpy array，再 to torch
                features_arr = np.stack(features, axis=0).astype(np.float32)  # shape [L, D]
                text_feat_tensor = torch.from_numpy(features_arr)  # torch.FloatTensor

                node_idxs_arr = np.array(node_idxs, dtype=np.int64)  # [L]
                node_idxs_tensor = torch.from_numpy(node_idxs_arr)
                device_idxs_arr = np.array(device_idxs, dtype=np.int64)
                device_idxs_tensor = torch.from_numpy(device_idxs_arr)

                # 转成 tensor 并保存
                self.samples.append({
                    'node_idxs': node_idxs_tensor,
                    'device_idxs': device_idxs_tensor,
                    'text_feat': text_feat_tensor,
                    'is_root': torch.tensor(root_labels, dtype=torch.long),
                    'is_true_fault': torch.tensor(true_labels, dtype=torch.long)
                })

            # 缓存到磁盘，下次直接加载：同时保存映射信息
            torch.save({
                'samples': self.samples,
                'idx_to_node': self.le_node.classes_.tolist(),
                'idx_to_device': self.le_dev.classes_.tolist()
            }, preproc_path)
            print(f"预处理完成，保存了 {len(self.samples)} 条数据在 {preproc_path}")
            elapsed = time.time() - start_time
            print(f"告警日志预处理耗时：{elapsed:.2f} 秒")
            # 加载完成后统计数据集中告警分布
            self.report_stats()


    def __len__(self):
        """返回样本数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
               按索引返回单条样本:
                    node_idxs: LongTensor([L])
                    device_idxs: LongTensor([L])
                    text_feat: FloatTensor([L, D])
                    is_root: LongTensor([L])
                    is_true_fault: LongTensor([L])
               """
        return self.samples[idx]

    def report_stats(self):
        """统计并打印告警类别分布"""
        # 统计包含根源告警的序列数量
        count_root = sum(1 for s in self.samples if s['is_root'].max() == 1)
        # 统计包含真实故障的序列数量
        count_true = sum(1 for s in self.samples if s['is_true_fault'].max() == 1)
        print(f"根源告警共有：{count_root} 条，衍生告警共有：{len(self.samples) - count_root} 条")
        print(f"真实故障共有：{count_true} 条，非真实根源告警有：{count_root - count_true} 条")