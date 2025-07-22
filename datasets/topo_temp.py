import json
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch.utils.data import Dataset

class TopologyDataset(Dataset):
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_path = json_path
        # --- 加载图数据 ---
        self.data = self.load_topology_dataset(json_path)
        # --- 节点特征维度 ---
        self.num_features = self.data.x.shape[1]
        # --- 在这之后，插入 PAD 节点 ---
        # 1) 在 x 的最前面插入一行全 0 作为 PAD 特征
        pad_row = torch.zeros((1, self.data.x.size(1)), dtype=self.data.x.dtype)
        self.data.x = torch.cat([pad_row, self.data.x], dim=0)

        # 2) 把所有 edge_index 的节点编号 +1
        self.data.edge_index = self.data.edge_index + 1

        # --- 接着保存 node_ids 列表时，同样要从 1 开始映射 ---
        with open(json_path, 'r', encoding='utf-8') as f:
            topo = json.load(f)
        # 这里 node_ids 还是原始名字列表，用来按相同顺序映射到 1..N
        self.node_ids = [n['id'] for n in topo['nodes']]
        # 你不需要把 PAD 放进 node_ids，node_ids 只存真实节点


    def load_topology_dataset(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            topo = json.load(f)

        nodes = topo['nodes']
        edges = topo['edges']

        node_ids = [n['id'] for n in nodes]
        id_to_index = {nid: i for i, nid in enumerate(node_ids)}

        # --- Node Features ---
        types = np.array([[n.get('type', 'unknown')] for n in nodes])
        enc = OneHotEncoder(sparse_output=False)
        type_features = enc.fit_transform(types)

        locations = np.array([n.get('location', [0.0, 0.0]) for n in nodes])
        node_features = np.concatenate([type_features, locations], axis=1)
        x = torch.tensor(node_features, dtype=torch.float)

        # --- Edge Index & Features ---
        edge_index = []
        edge_attrs = []
        for edge in edges:
            src = id_to_index.get(edge['source'])
            tgt = id_to_index.get(edge['target'])
            if src is None or tgt is None:
                print(f"跳过边：{edge['source']} -> {edge['target']}")
                continue
            edge_index.append([src, tgt])

            # 边特征：带宽、距离、延迟
            bandwidth = self.parse_bandwidth(edge.get('bandwidth', '0G'))
            distance = edge.get('distance_km', 0.0)
            latency = edge.get('latency_ms', 0.0)
            edge_attrs.append([bandwidth, distance, latency])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def parse_bandwidth(self, bw_str):
        if isinstance(bw_str, (int, float)):
            return float(bw_str)
        bw_str = bw_str.upper().strip()
        if bw_str.endswith("G"):
            return float(bw_str[:-1]) * 1000
        elif bw_str.endswith("M"):
            return float(bw_str[:-1])
        return 0.0

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1  # 只有一个图，返回 1
