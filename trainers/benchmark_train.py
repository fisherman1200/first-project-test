# coding: utf-8
"""训练并评估基准模型

使用 AlarmDataset 预处理的序列特征，
实现 CONAD、LogBERT 等模型的训练与评估。
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from models.benchmark_models import (
    CONAD, LogBERT, LogGD, DeepTraLog,
    Graphormer, GraphMAE, DistilBERTGraph,
)
from trainers.anomaly_train import load_sequence_features, evaluate_preds


class SequenceDataset(Dataset):
    """将 numpy 特征封装为 PyTorch Dataset"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_model(name: str, input_dim: int) -> nn.Module:
    """根据名称实例化模型"""
    name = name.lower()
    if name == 'conad':
        return CONAD(input_dim)
    if name == 'logbert':
        return LogBERT(input_dim)
    if name == 'loggd':
        return LogGD(input_dim)
    if name == 'deeptralog':
        return DeepTraLog(input_dim)
    if name == 'graphormer':
        return Graphormer(input_dim)
    if name == 'graphmae':
        return GraphMAE(input_dim)
    if name == 'distilbertgraph':
        return DistilBERTGraph(gnn_channels=input_dim)
    raise ValueError(f'未知模型: {name}')


def train_epoch(model, loader, optimizer, device):
    """训练一个 Epoch，并返回损失与评估指标"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    preds, labels, scores = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
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
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float('nan')
    return total_loss / len(loader.dataset), acc, prec, rec, f1, auc


def eval_epoch(model, loader, device):
    """在验证或测试集上评估模型"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    preds, labels, scores = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.extend(out.argmax(dim=1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            scores.extend(prob.cpu().tolist())
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float('nan')
    return total_loss / len(loader.dataset), acc, prec, rec, f1, auc, preds, labels, scores


def run_training(cfg, model_name: str, task: str = 'root', epochs: int = 5):
    """训练指定模型并返回测试指标"""
    preproc = 'data/processed/processed_alarm_sequences_v1.pt'
    X, y_root, y_true = load_sequence_features(cfg, preproc)
    if task == 'root':
        y = y_root
    else:
        mask = y_root == 1
        X = X[mask]
        y = y_true[mask]
        if len(X) == 0:
            print('数据中没有根源告警，无法进行真实故障检测')
            return None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg.data.num_val + cfg.data.num_test, random_state=42
    )
    val_ratio = cfg.data.num_val / (cfg.data.num_val + cfg.data.num_test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=42
    )
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = 0.0
    best_state = model.state_dict()
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, device)
        _, acc, prec, rec, f1, auc, _, _, _ = eval_epoch(model, val_loader, device)
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    loss, acc, prec, rec, f1, auc, preds, labels, scores = eval_epoch(model, test_loader, device)
    evaluate_preds(labels, preds, scores)
    print(
        f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
    )
    return {'loss': loss, 'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


def train_benchmark(cfg, model_name: str, epochs: int = 5):
    """同时在根因检测与真实故障检测任务上训练评估"""
    print(f"===== 模型: {model_name} | 根因检测 =====")
    res_root = run_training(cfg, model_name, 'root', epochs)
    print(f"===== 模型: {model_name} | 真实故障检测 =====")
    res_true = run_training(cfg, model_name, 'true', epochs)
    return res_root, res_true
