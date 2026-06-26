"""
可视化工具
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_fusion(sources, fused, gt=None, save_path=None):
    """
    可视化融合结果

    Args:
        sources: 源图像列表 (numpy 或 tensor)
        fused: 融合图像
        gt: Ground Truth (可选)
        save_path: 保存路径 (可选)

    Returns:
        fig: matplotlib figure
    """
    import torch
    if isinstance(sources[0], torch.Tensor):
        sources = [s.cpu().numpy() for s in sources]
    if isinstance(fused, torch.Tensor):
        fused = fused.cpu().numpy()
    if gt is not None and isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    # 转换为 HWC 格式
    def to_hwc(img):
        if img.ndim == 4:
            img = img[0]  # 取第一个样本
        if img.shape[0] == 3:  # CHW -> HWC
            img = img.transpose(1, 2, 0)
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    sources_hwc = [to_hwc(s) for s in sources]
    fused_hwc = to_hwc(fused)

    # 创建图形
    n_cols = min(5, len(sources) + 1 + (1 if gt else 0))
    n_rows = 1

    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    if n_rows == 1:
        axes = [axes]

    # 显示源图像 (前 4 张)
    for i, src in enumerate(sources_hwc[:4]):
        if i < len(axes) - 1 - (1 if gt else 0):
            axes[i].imshow(src)
            axes[i].set_title(f'Source {i+1}')
            axes[i].axis('off')

    # 显示融合结果
    ax_idx = min(4, len(sources_hwc))
    axes[ax_idx].imshow(fused_hwc)
    axes[ax_idx].set_title('Fused')
    axes[ax_idx].axis('off')

    # 显示 GT
    if gt is not None:
        gt_hwc = to_hwc(gt)
        axes[-1].imshow(gt_hwc)
        axes[-1].set_title('Ground Truth')
        axes[-1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.close()

    return fig


def plot_metrics(train_losses, val_losses, save_path=None):
    """
    绘制训练/验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(val_losses, 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.close()

    return fig


def plot_metrics_radar(metrics_dict, save_path=None):
    """
    绘制指标雷达图

    Args:
        metrics_dict: {model_name: {metric_name: value}}
        save_path: 保存路径
    """
    # 获取所有指标
    all_metrics = set()
    for m in metrics_dict.values():
        all_metrics.update(m.keys())
    all_metrics = list(all_metrics)

    # 归一化
    max_vals = {m: max(metrics_dict[n].get(m, 0) for n in metrics_dict)
                for m in all_metrics}

    angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_dict)))

    for (name, metrics), color in zip(metrics_dict.items(), colors):
        values = [metrics.get(m, 0) / max_vals[m] for m in all_metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_metrics)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Metrics Comparison', fontsize=14)
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def create_comparison_grid(images_dict, titles, save_path=None):
    """
    创建对比网格图

    Args:
        images_dict: {name: image}
        titles: 标题列表
        save_path: 保存路径
    """
    n = len(images_dict)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (name, img) in enumerate(images_dict.items()):
        import torch
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        axes[idx].imshow(img)
        axes[idx].set_title(titles[idx] if idx < len(titles) else name)
        axes[idx].axis('off')

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig
