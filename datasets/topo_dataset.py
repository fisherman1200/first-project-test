import json, torch, os
from torch_geometric.data import Dataset

class TopologyDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            topo = json.load(f)

        # 根据类型分开收集 node_ids
        nodes_by_type = {'core': [], 'agg': [], 'access': []}
        self.node_ids = nodes_by_type['core'] + nodes_by_type['agg'] + nodes_by_type['access']
        for n in topo['nodes']:
            nodes_by_type[n['type']].append(n['id'])

        self.node_map = {
            t: {nid: i for i, nid in enumerate(nodes_by_type[t])}
            for t in nodes_by_type
        }

        # 零向量占位（3维）
        zero_vec = [0.0, 0.0, 0.0]
        self.feature_dim = len(zero_vec)

        self.x_dict = {
            t: torch.tensor([zero_vec]*len(nodes_by_type[t]), dtype=torch.float)
            for t in nodes_by_type
        }

        # 构建异构边索引与边特征
        self.edge_index_dict = {}
        self.edge_attr_dict = {}
        for e in topo['edges']:
            s, t = e['source'], e['target']
            # 找类型
            s_type = next(n['type'] for n in topo['nodes'] if n['id']==s)
            t_type = next(n['type'] for n in topo['nodes'] if n['id']==t)
            rel = (s_type, 'to', t_type)
            si = self.node_map[s_type].get(s, None)
            ti = self.node_map[t_type].get(t, None)
            if si is None or ti is None:
                continue

            # 解析边特征
            bw = float(e['bandwidth'].rstrip('G')) if 'G' in e['bandwidth'] else float(e['bandwidth'])
            dist = float(e.get('distance_km', 0.0))
            delay = float(e.get('latency_ms', 0.0))

            self.edge_index_dict.setdefault(rel, []).append([si, ti])
            self.edge_attr_dict.setdefault(rel, []).append([bw, dist, delay])

        # 转 tensor
        for rel, idxs in self.edge_index_dict.items():
            self.edge_index_dict[rel] = torch.tensor(idxs, dtype=torch.long).t().contiguous()
            self.edge_attr_dict[rel]  = torch.tensor(self.edge_attr_dict[rel], dtype=torch.float)

    def len(self):
        return 1

    def get(self, idx):
        return self.x_dict, self.edge_index_dict, self.edge_attr_dict
