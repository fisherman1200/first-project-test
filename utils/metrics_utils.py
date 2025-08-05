import json
import os
from typing import List, Dict, Any
from utils.path_utils import get_output_dir, get_run_timestamp


class MetricsLogger:
    """
    MetricsLogger 用于统一管理训练过程中的各种指标。

    用法：
        logger = MetricsLogger(keys=['root', 'true', 'acc'])
        for epoch in ...:
            # 计算 avg_metrics = {'root': avg_root, 'true': avg_true, 'acc': avg_acc}
            logger.add(epoch, avg_metrics)
        # 保存结果到 data/raw/metrics_data/ 时间戳目录下，文件名前缀可自定义
        logger.save(prefix='my_model')
    """
    def __init__(self, keys: List[str]):
        # keys: 指标名列表
        self.keys = keys
        # 初始化一个空 dict: key -> list of values
        self.data: Dict[str, List[Any]] = {k: [] for k in keys}
        self.epochs: List[int] = []

    def add(self, epoch: int, metrics: Dict[str, Any]):
        """
        添加当前 epoch 的所有指标，metrics 必须包含初始化的 keys。
        """
        self.epochs.append(epoch)
        for k in self.keys:
            if k not in metrics:
                raise KeyError(f"Metrics for key '{k}' missing in metrics dict.")
            self.data[k].append(metrics[k])

    def save(self, prefix: str = "metrics_data") -> str:
        """
        将收集到的指标保存为 JSON 文件，包含 epochs 列表和每个 key 对应的值。
        参数:
            prefix: 输出文件名前缀，便于区分不同模型/实验。

        返回:
            输出文件的完整路径。
        """
        # 生成默认输出目录和文件名，目录带有统一的运行时间戳
        default_dir = get_output_dir('data', 'raw', 'metrics_data')
        filename = f'{prefix}.json'
        output_path = os.path.join(default_dir, filename)

        os.makedirs(default_dir, exist_ok=True)
        out = {k: self.data[k] for k in self.keys}
        out['epoch'] = self.epochs
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        # 返回输出路径，方便外部记录
        return output_path