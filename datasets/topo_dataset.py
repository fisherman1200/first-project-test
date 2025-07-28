import json, torch
from torch_geometric.data import Dataset

class TopologyDataset(Dataset):
    """
        TopologyDataset 用于加载网络拓扑图，并构建适用于异构图神经网络 (HeteroGNN) 的输入格式。

        输入:
            json_path (str): 包含网络拓扑信息的 JSON 文件路径。
                JSON 文件结构示例：
                {
                    "nodes": [
                        {"id": "node1", "type": "core"},
                        {"id": "node2", "type": "agg"},
                        ...,
                    ],
                    "edges": [
                        {"source": "node1", "target": "node2", "bandwidth": "10G", "distance_km": 5.0, "latency_ms": 1.0},
                        ...,
                    ]
                }

        属性:
            node_ids (List[str]): 按顺序拼接 core、agg、access 类型节点的 ID 列表。
            node_map (Dict[str, Dict[str, int]]): 三个子字典，key 为节点类型('core','agg','access')，
                value 为 {node_id: 索引} 对应关系，索引从 0 开始。
            feature_dim (int): 节点特征维度，本实现中固定为 3（占位零向量长度）。
            x_dict (Dict[str, Tensor]): 每个节点类型对应的节点特征矩阵，形状为 [num_nodes_type, feature_dim]。
            edge_index_dict (Dict[Tuple[str,str,str], Tensor]): 异构边索引字典，key 为 (src_type, 'to', tgt_type)，
                value 为长整型张量 shape=[2, num_edges]。
            edge_attr_dict (Dict[Tuple[str,str,str], Tensor]): 异构边特征字典，key 同上，
                value 为浮点型张量 shape=[num_edges, 3]，分别对应带宽、距离、时延。

        方法:
            len(): 返回数据集样本数（本类只返回 1 次完整图）。
            get(idx): 返回 (x_dict, edge_index_dict, edge_attr_dict)，忽略 idx 参数。

        使用方法:
            dataset = TopologyDataset('data/topo_graph.json')
            x_dict, edge_idx_dict, edge_attr_dict = dataset[0]
            # x_dict['core'] => Tensor([num_core_nodes, 3])
            # edge_idx_dict[('core','to','agg')] => Tensor(shape=[2, num_edges])
            # edge_attr_dict[('core','to','agg')] => Tensor(shape=[num_edges,3])
        """
    def __init__(self, json_path):
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            topo = json.load(f)

        # 按节点类型收集所有节点 ID
        nodes_by_type = {'core': [], 'agg': [], 'access': []}
        for n in topo['nodes']:
            nodes_by_type[n['type']].append(n['id'])
        self.node_ids = nodes_by_type['core'] + nodes_by_type['agg'] + nodes_by_type['access']

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
            # 找到对应的节点类型
            s_type = next(n['type'] for n in topo['nodes'] if n['id'] == s)
            t_type = next(n['type'] for n in topo['nodes'] if n['id'] == t)
            rel = (s_type, 'to', t_type)
            # 获取在该类型子集里的索引，如果某个 ID 不在映射中则跳过
            src_idx = self.node_map[s_type].get(s, None)
            tgt_idx = self.node_map[t_type].get(t, None)
            if src_idx is None or tgt_idx is None:
                continue

            # 解析边特征
            bw = float(e['bandwidth'].rstrip('G')) if 'G' in e['bandwidth'] else float(e['bandwidth'])
            dist = float(e.get('distance_km', 0.0))
            delay = float(e.get('latency_ms', 0.0))

            self.edge_index_dict.setdefault(rel, []).append([src_idx, tgt_idx])
            self.edge_attr_dict.setdefault(rel, []).append([bw, dist, delay])

        # 转 tensor
        for rel, idxs in self.edge_index_dict.items():
            self.edge_index_dict[rel] = torch.tensor(idxs, dtype=torch.long).t().contiguous()
            self.edge_attr_dict[rel]  = torch.tensor(self.edge_attr_dict[rel], dtype=torch.float)

    def len(self):
        """返回数据集样本数，本实现中仅有一张完整的拓扑图。"""
        return 1

    def get(self, idx):
        """
                返回第 idx 张图的节点与边信息：
                    x_dict: 异构节点特征（torch.Tensor）
                    edge_index_dict: 异构边索引（torch.Tensor）
                    edge_attr_dict: 异构边属性（torch.Tensor）
                忽略 idx 值，因为只有一张图。
                """
        return self.x_dict, self.edge_index_dict, self.edge_attr_dict
