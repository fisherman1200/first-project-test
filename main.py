from trainers.train import train_model
from utils.config import load_config
from trainers.algorithm_train import train_algorithm
import torch

def main():
    # 可选算法列表
    VALID_ALGOS = [
        'my', 'pca', 'isoforest', 'ocsvm',
        'conad', 'logbert', 'loggd', 'deeptralog',
        'graphormer', 'graphmae', 'distilbertgraph'
    ]

    # 2. 运行时交互式输入
    print("可选算法：", ", ".join(VALID_ALGOS))
    algo = input("请输入算法（默认 my）：") or "my"
    if algo not in VALID_ALGOS:
        print(f"无效算法 '{algo}'，已切回默认 my。")
        algo = "my"

    # 3. 加载配置
    cfg = load_config('configs/config.yaml')

    # 4. 根据输入分支
    if algo == 'my':
        train_model(cfg)
    else:
        train_algorithm(cfg, algo)

if __name__ == '__main__':
    main()
