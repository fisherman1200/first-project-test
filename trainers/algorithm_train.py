"""统一的算法训练接口

此模块根据算法名称自动选择对应的训练流程，
支持 PCA/IsolationForest/One-Class SVM 等异常检测算法，
以及若干基准深度学习模型。
"""

from trainers.anomaly_train import train_anomaly
from trainers.benchmark_runner import train_benchmark

# 归一化名称，便于匹配
_ANOMALY_ALGOS = {"pca", "isoforest", "ocsvm"}


def train_algorithm(cfg, algo: str):
    """根据算法名称执行训练

    参数:
        cfg: 配置对象
        algo: 算法名称，大小写不敏感
    """
    algo = algo.lower()
    if algo in _ANOMALY_ALGOS:
        return train_anomaly(cfg, method=algo)
    else:
        return train_benchmark(cfg, algo)


__all__ = ["train_algorithm"]