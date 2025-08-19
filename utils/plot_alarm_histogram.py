#!/usr/bin/env python3
"""
统计告警数据的时间分布并绘制直方图。

默认从 ``data/alarms.json`` 读取告警列表，每条记录需包含 ``timestamp`` 字段，
格式示例：``YYYY-mm-ddTHH:MM:SS``。

使用示例：
    python utils/plot_alarm_histogram.py --freq 1H --output alarm_hist.pdf

参数说明：
    --freq/-f   时间分箱间隔，使用 Pandas 频率字符串，例如 ``30min``、``1H`` 等；
    --output/-o 输出文件路径，默认保存到 ``data/processed/alarms/`` 下。
"""

import json
import argparse
import os
from typing import Sequence

import pandas as pd
import matplotlib.pyplot as plt

# 全局 Matplotlib 配置，与项目中其他绘图脚本保持一致
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


def _load_timestamps(json_path: str) -> pd.Series:
    """读取 JSON 文件并返回 Pandas 时间序列"""
    with open(json_path, "r", encoding="gbk") as f:   # 更稳的编码
        data = json.load(f)
    times = [item["timestamp"] for item in data]
    # 关键：允许带微秒等 ISO8601 变体
    return pd.to_datetime(times, format="ISO8601")  # 或者 format="mixed"


def plot_alarm_histogram(timestamps: Sequence[pd.Timestamp], freq: str, output_path: str) -> None:
    """按指定间隔统计告警数量，并绘制直方图"""
    ts_series = pd.to_datetime(list(timestamps))
    # 使用 floor 将时间对齐到指定频率，然后统计数量
    counts = ts_series.floor(freq).value_counts().sort_index()

    fig, ax = plt.subplots()
    # 使用条形图展示，每个条的宽度与分箱间隔一致
    width = pd.to_timedelta(freq)
    ax.bar(counts.index, counts.values, width=width)
    ax.set_xlabel(r"\textbf{Time}")
    ax.set_ylabel(r"\textbf{Number}")
    ax.set_title(r"\textbf{Alarm Time Distribution}")
    ax.grid(linestyle="--", alpha=0.5)
    fig.autofmt_xdate()  # 自动旋转日期标签，防止重叠
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved alarm histogram to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计并绘制告警数据时间分布")
    parser.add_argument(
        "--input", "-i", type=str, default="data/alarms.json",
        help="包含告警数据的 JSON 文件路径"
    )
    parser.add_argument(
        "--freq", "-f", type=str, default="1H",
        help="时间分箱间隔，使用 Pandas 频率字符串"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出图像文件路径"
    )
    args = parser.parse_args()

    timestamps = _load_timestamps(args.input)

    # 默认输出目录
    default_dir = os.path.join("data", "processed", "alarms")
    os.makedirs(default_dir, exist_ok=True)
    if args.output is None:
        output_path = os.path.join(default_dir, "alarm_histogram.pdf")
    else:
        # 若仅提供文件名，则放入默认目录
        output_path = args.output if os.path.dirname(args.output) else os.path.join(default_dir, args.output)

    plot_alarm_histogram(timestamps, args.freq, output_path)


if __name__ == "__main__":
    main()