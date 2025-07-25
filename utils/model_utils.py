import os
import torch
from datetime import datetime


class ModelSaver:
    """
    ModelSaver 用于在一次训练运行内统一保存最佳模型权重。

    运行时刻初始化时会创建一个唯一目录：
        data/processed/model_<timestamp>/
    保存时只生成两个文件：
        best_gnn.pth     # GNNTransformer 最佳权重
        best_at.pth      # AlarmTransformer 最佳权重

    用法：
        saver = ModelSaver(base_dir='data/processed')
        # 训练过程中，当发现新的最优时：
        saver.save_best(gnn=gnn_model, at=alarm_transformer)
    """

    def __init__(self, base_dir: str = 'data/processed'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(base_dir, f'model_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        # 临时存储上次保存的 paths
        self.saved_paths = {}

    def save_best(self, **model_dict):
        """
        仅保存最佳模型：覆盖写入 best_<name>.pth
        示例：
            save_best(gnn=gnn, at=alarm_transformer)
        会在 output_dir 下生成：
            best_gnn.pth
            best_at.pth
        """
        for name, model in model_dict.items():
            if not hasattr(model, 'state_dict'):
                raise ValueError(f"Object '{name}' has no state_dict() method.")
            filename = f"best_{name}.pth"
            path = os.path.join(self.output_dir, filename)
            torch.save(model.state_dict(), path)
            self.saved_paths[name] = path
        return self.saved_paths


