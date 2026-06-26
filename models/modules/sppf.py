"""
SPPF 模块 - 快速空间金字塔池化

Spatial Pyramid Pooling - Fast, 来自 YOLOv5 的高效多尺度特征提取模块。
"""

import torch
import torch.nn as nn


class SPPF(nn.Module):
    """
    快速空间金字塔池化模块

    通过串联多个相同大小的最大池化操作，
    实现多尺度特征提取，同时保持高效的计算速度。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 池化核大小 (默认 5)

    Structure:
        Input -> 1x1 Conv -> [5x5 MaxPool] x3 (串联) -> Concat -> 1x1 Conv -> Output

    相比传统 SPP 的优势:
        - 传统 SPP: 并行使用多个不同大小的池化核 (如 5x5, 9x9, 13x13)
        - SPPF: 串联使用相同大小的池化核，等效于更大的感受野
        - 计算效率更高，更容易部署

    感受野等效:
        - 1 个 5x5 池化：感受野 5x5
        - 2 个串联：感受野 9x9
        - 3 个串联：感受野 13x13
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        # 隐藏层通道数设为输入通道的一半，用于降维减少计算量
        hidden_dim = in_channels // 2

        # 通道压缩
        # 第一个 1x1 卷积：降低通道维数，减少后续池化的计算量
        self.cv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)   # 批归一化，加速收敛

        # 输出投影
        # 第二个 1x1 卷积：将拼接后的多尺度特征（通道数为 hidden_dim*4）投影到输出通道数
        self.cv2 = nn.Conv2d(hidden_dim * 4, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 最大池化
        # 最大池化层：kernel_size 为指定值，padding 保持特征图尺寸不变
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1,
                              padding=kernel_size//2)
        # 使用 ReLU6 激活函数（输出范围 [0,6]，适合低精度推理）
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)

        Returns:
            多尺度融合特征图 (B, out_channels, H, W)
        """
        # 通道压缩
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # 串联池化
        # 每次池化保持空间尺寸不变（stride=1, padding=k//2）
        y1 = self.m(x)      # 5x5 等效
        y2 = self.m(y1)     # 9x9 等效
        y3 = self.m(y2)     # 13x13 等效

        # 拼接多尺度特征
        # 将原始特征、一次池化、二次池化、三次池化的结果在通道维度拼接
        # 拼接后通道数 = hidden_dim * 4
        out = torch.cat([x, y1, y2, y3], dim=1)

        # 通道投影
        # 使用 1x1 卷积将通道数降为 out_channels，同时融合多尺度信息
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.act(out)

        return out

    def extra_repr(self) -> str:
        # 用于打印模块信息时显示额外参数
        return f'kernel_size=5, expansion=4'


class SPP(nn.Module):
    """
    传统空间金字塔池化模块 (保留用于对比)

    使用并行不同大小的池化核。
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 9, 13]):
        super().__init__()
        # 隐藏层通道数：输入通道的一半
        hidden_dim = in_channels // 2

        # 1x1 卷积降维
        self.cv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        # 输出投影：拼接后的通道数为 hidden_dim * (len(kernel_sizes)+1)
        # 因为要拼接原始特征和每个池化结果
        self.cv2 = nn.Conv2d(hidden_dim * (len(kernel_sizes) + 1), out_channels, 1, bias=False)

        # 并行使用多个不同 kernel_size 的最大池化层
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
            for k in kernel_sizes
        ])

    def forward(self, x):
        # 降维
        x = self.cv1(x)
        # 对每个池化层分别处理
        pools = [m(x) for m in self.m]
        # 拼接原始特征和所有池化结果
        out = torch.cat([x] + pools, dim=1)
        # 输出投影，恢复指定通道数
        return self.cv2(out)
