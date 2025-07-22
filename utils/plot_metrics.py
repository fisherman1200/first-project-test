#!/usr/bin/env python3
"""
plot_metrics.py

通用绘图脚本，根据指定 JSON 文件里的多项训练指标（如 loss, accuracy, lr 等），
按用户指定的 key 列表生成对应的曲线图。

使用方法：根目录终端运行
    python .\utils\plot_metrics.py \
      --input metrics.json \
      --keys root,true \
      --output loss_curves.pdf

    python .\utils\plot_metrics.py --keys root,true

如果不指定 --output，将默认生成在 data/processed/ 下，
文件名为 "metrics_<YYYYMMDD_HHMMSS>.pdf"；
如果只指定文件名（不含路径），也会自动放到 data/processed/ 下。

输入 JSON 文件格式示例：
{
  "root": [0.5, 0.4, 0.3, ...],
  "true": [1.2, 0.9, 0.7, ...],
  "acc": [0.60, 0.65, 0.70, ...],
  "lr": [0.001, 0.001, 0.0005, ...]
}
"""
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 全局 Matplotlib 配置：LaTeX 字体、字号、线型、矢量输出
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "savefig.dpi": 300,
})


def plot_selected(data: dict, keys: list, output_path: str):
    # 检查 keys 是否存在且长度一致
    length = None
    for k in keys:
        if k not in data:
            raise KeyError(f"Key '{k}' not found in input data.")
        if length is None:
            length = len(data[k])
        elif len(data[k]) != length:
            raise ValueError("Selected metrics must have the same length.")

    epochs = list(range(1, length + 1))
    plt.figure()
    for k in keys:
        values = data[k]
        plt.plot(epochs, values, marker='o', label=rf'\textbf{{{k}}}')

    plt.xlabel(r'\textbf{Epoch}')
    plt.ylabel(r'\textbf{Value}')
    title = " \\& ".join(keys) + " Curve"
    plt.title(rf'\textbf{{{title}}}')
    plt.legend(loc='best', frameon=False)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot of {keys} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot selected training metrics.")
    parser.add_argument(
        '--input', '-i', type=str, default='data/metrics.json',
        help='Path to JSON file containing metrics'
    )
    parser.add_argument(
        '--keys', '-k', type=str, required=True,
        help='Comma-separated list of keys to plot (e.g., root,true)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output figure file (e.g., loss_curves.pdf)'
    )
    args = parser.parse_args()

    # 读取 JSON 数据
    with open(args.input, 'r') as f:
        data = json.load(f)
    # 解析 keys
    keys = [key.strip() for key in args.keys.split(',') if key.strip()]
    if not keys:
        raise ValueError('No keys specified for plotting.')

    # 生成默认输出目录和文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_dir = os.path.join('data', 'processed')
    if args.output is None:
        filename = f'metrics_{timestamp}.pdf'
    else:
        filename = args.output
    # 如果用户只给了文件名（没有路径），则放到 default_dir
    if not os.path.dirname(filename):
        output_path = os.path.join(default_dir, filename)
    else:
        output_path = filename

    # 绘图
    plot_selected(data, keys, output_path)


if __name__ == '__main__':
    main()
