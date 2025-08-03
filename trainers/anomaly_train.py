import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datasets.alarm_dataset import AlarmDataset
from collections import Counter


def load_sequence_features(cfg, preproc_path='data/processed/processed_alarm_sequences_v1.pt'):
    """加载序列级特征和标签

    每个序列取平均特征表示，返回：
        - X: ndarray (N, D)
        - y_root: 是否为根源告警
        - y_true: 根源告警是否为真实故障
    """
    ds = AlarmDataset(
        cfg.data.alarm_path,
        node_id_map=None,
        max_len=cfg.data.max_len,
        window_milliseconds=cfg.data.window_milliseconds,
        step_milliseconds=cfg.data.step_milliseconds,
        preproc_path=preproc_path,
    )
    feats = []
    root_labels = []
    true_labels = []
    for sample in ds:
        # 平均池化得到序列特征向量
        seq_feat = sample['text_feat'].mean(dim=0).numpy()
        root_flag = int(sample['is_root'].max().item())
        true_flag = int(sample['is_true_fault'].max().item())
        feats.append(seq_feat)
        root_labels.append(root_flag)
        true_labels.append(true_flag)
    root_arr = np.array(root_labels)
    true_arr = np.array(true_labels)
    print("== 标签统计 ==")
    print("Root:", Counter(root_arr.tolist()))
    print("True:", Counter(true_arr.tolist()))
    return np.stack(feats), root_arr, true_arr


def evaluate_preds(y_true, y_pred, scores=None):
    """计算并打印常用指标，可选传入连续得分计算 AUC"""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if scores is not None and len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, scores)
        print(
            f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
        )
    else:
        print(
            f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )


def train_anomaly(cfg, method='pca'):
    """运行经典异常检测算法并评估两种标签"""

    X, y_root, y_true = load_sequence_features(cfg)

    # -------- 根因检测 --------
    X_train_root = X[y_root == 0]  # 仅使用非根源样本训练
    X_test_root = X
    y_test_root = y_root

    def fit_predict_scores(X_tr, X_te):
        """根据算法类型训练并返回预测结果与得分"""
        if method == 'pca':
            print("使用主成分分析 (PCA) 进行异常检测")
            mdl = PCA(n_components=min(10, X_tr.shape[1]))
            mdl.fit(X_tr)
            recon_tr = mdl.inverse_transform(mdl.transform(X_tr))
            thresh = np.percentile(np.linalg.norm(X_tr - recon_tr, axis=1), 95)
            recon_te = mdl.inverse_transform(mdl.transform(X_te))
            scores = np.linalg.norm(X_te - recon_te, axis=1)
            preds = (scores > thresh).astype(int)
        elif method == 'isoforest':
            print("使用 Isolation Forest 进行异常检测")
            mdl = IsolationForest(contamination=0.1, random_state=42)
            mdl.fit(X_tr)
            scores = -mdl.score_samples(X_te)
            preds = mdl.predict(X_te)
            preds = (preds == -1).astype(int)
        elif method == 'ocsvm':
            print("使用 One-Class SVM 进行异常检测")
            mdl = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            mdl.fit(X_tr)
            scores = -mdl.score_samples(X_te)
            preds = mdl.predict(X_te)
            preds = (preds == -1).astype(int)
        else:
            raise ValueError(f"未知算法: {method}")
        return preds, scores

    # 根因检测评估
    print("root任务：")
    preds_root, scores_root = fit_predict_scores(X_train_root, X_test_root)
    evaluate_preds(y_test_root, preds_root, scores_root)

    # -------- 真实故障检测 --------
    mask_root = y_root == 1
    X_root = X[mask_root]
    y_true_root = y_true[mask_root]
    if len(X_root) == 0:
        print("数据中没有根源告警，跳过真实故障检测")
        return
    X_train_true = X_root[y_true_root == 0]
    if len(X_train_true) == 0:
        print("没有非真实故障样本可用于训练，跳过真实故障检测")
        return
    print("true任务：")
    preds_true, scores_true = fit_predict_scores(X_train_true, X_root)
    evaluate_preds(y_true_root, preds_true, scores_true)