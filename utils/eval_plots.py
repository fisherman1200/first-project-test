"""
根据预测结果绘制混淆矩阵和 ROC 曲线，风格遵循 LaTeX 论文排版要求。
"""

import os

import numpy as np

from utils.path_utils import get_output_dir
from typing import Sequence, Tuple, Any

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (4.5, 4),
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "savefig.dpi": 300,
})


def _make_output_path(prefix: str, suffix: str) -> str:
    directory = get_output_dir("data", "processed", "eval")
    filename = f"{prefix}.{suffix}"
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def plot_confusion(labels: Sequence[int], preds: Sequence[int],
                   class_names: Tuple[str, str], prefix: str) -> str:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel(r"\textbf{Predicted}")
    ax.set_ylabel(r"\textbf{True}")
    ax.set_xticks([0, 1], labels=[rf"\textbf{{{n}}}" for n in class_names])
    ax.set_yticks([0, 1], labels=[rf"\textbf{{{n}}}" for n in class_names])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], va="center", ha="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = _make_output_path(prefix, "pdf")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")
    return out_path


def plot_roc(labels: Sequence[int], probs: Sequence[float], prefix: str) -> tuple[str, Any]:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    print("最佳阈值（Youden）:", best_threshold)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=rf"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel(r"\textbf{FPR}")
    ax.set_ylabel(r"\textbf{TPR}")
    ax.set_title(r"\textbf{ROC Curve}")
    ax.legend(frameon=False)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = _make_output_path(prefix + "_roc", "pdf")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved ROC curve to {out_path}")
    return out_path, best_threshold


__all__ = ["plot_confusion", "plot_roc"]