"""基准模型训练流程封装

包含数据集封装、单轮训练评估以及完整训练函数，
用于在根因检测和真故障检测任务上评估不同模型。
"""

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from datasets.topo_dataset import TopologyDataset
from models.benchmark_models import get_benchmark_model
from trainers.anomaly_train import load_sequence_features, evaluate_preds


class SequenceDataset(Dataset):
    """简单的序列特征数据集"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_edge_index(cfg, device):
    """根据拓扑文件构建全局边索引"""
    topo_ds = TopologyDataset(cfg.data.topo_path)
    x_dict, edge_index_dict, _ = topo_ds[0]
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    offset = {
        "core": 0,
        "agg": x_dict["core"].size(0),
        "access": x_dict["core"].size(0) + x_dict["agg"].size(0),
    }
    edges = []
    for (src_type, _, tgt_type), eidx in edge_index_dict.items():
        src = eidx[0] + offset[src_type]
        tgt = eidx[1] + offset[tgt_type]
        edges.append(torch.stack([src, tgt], dim=0))
    return torch.cat(edges, dim=1)


def run_epoch(model, loader, device, optimizer=None):
    """执行一轮训练或评估"""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    preds, labels, scores = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        prob = torch.softmax(out, dim=1)[:, 1]
        preds.extend(out.argmax(dim=1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        scores.extend(prob.detach().cpu().tolist())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    return (
        total_loss / len(loader.dataset),
        acc,
        prec,
        rec,
        f1,
        auc,
        preds,
        labels,
        scores,
    )


def train_once(cfg, model_name: str, task: str = "root", epochs: int = 5):
    """训练指定模型，返回测试集指标"""
    preproc = "data/processed/processed_alarm_sequences_v1.pt"
    X, y_root, y_true = load_sequence_features(cfg, preproc)
    if task == "root":
        y = y_root
    else:
        mask = y_root == 1
        X = X[mask]
        y = y_true[mask]
        if len(X) == 0:
            print("数据中没有根源告警，无法进行真实故障检测")
            return None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg.data.num_val + cfg.data.num_test, random_state=42
    )
    val_ratio = cfg.data.num_val / (cfg.data.num_val + cfg.data.num_test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=42
    )

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train), batch_size=cfg.data.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val), batch_size=cfg.data.batch_size
    )
    test_loader = DataLoader(
        SequenceDataset(X_test, y_test), batch_size=cfg.data.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = build_edge_index(cfg, device)
    model = get_benchmark_model(model_name, X.shape[1], edge_index).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1, best_state = 0.0, model.state_dict()
    for _ in range(epochs):
        run_epoch(model, train_loader, device, optimizer)
        _, _, _, _, f1, _, _, _, _ = run_epoch(model, val_loader, device)
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    loss, acc, prec, rec, f1, auc, preds, labels, scores = run_epoch(
        model, test_loader, device
    )
    evaluate_preds(labels, preds, scores)
    return {
        "loss": loss,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


def train_benchmark(cfg, model_name: str, epochs: int = 5):
    """同时在根因检测与真实故障检测任务上训练评估"""
    print(f"===== 模型: {model_name} | 根因检测 =====")
    res_root = train_once(cfg, model_name, "root", epochs)
    print(f"===== 模型: {model_name} | 真实故障检测 =====")
    res_true = train_once(cfg, model_name, "true", epochs)
    return res_root, res_true