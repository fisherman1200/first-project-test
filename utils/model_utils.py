import os
import torch
from datetime import datetime

class ModelSaver:
    """
    ModelSaver 用于按时间戳自动保存 PyTorch 模型权重。

    用法：
        saver = ModelSaver(output_dir='data/processed/model')
        saver.save(gnn=gnn_model, at=transformer_model)
    """
    def __init__(self, output_dir: str = 'data/processed/model'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, **model_dict):
        """
        接收任意命名的模型状态字典，并按“<name>_<timestamp>.pth”格式保存。

        示例：
            save(gnn=gnn, at=alarm_transformer)
        将生成：
            gnn_20250723_153045.pth
            at_20250723_153045.pth
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_paths = {}
        for name, model in model_dict.items():
            if hasattr(model, 'state_dict'):
                filename = f"{name}_{timestamp}.pth"
                path = os.path.join(self.output_dir, filename)
                torch.save(model.state_dict(), path)
                saved_paths[name] = path
            else:
                raise ValueError(f"Object '{name}' has no state_dict() method.")
        return saved_paths

