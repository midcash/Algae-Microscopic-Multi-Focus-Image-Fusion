"""
深度可分离卷积模块

基于 MobileNet 思想，使用深度可分离卷积替代标准卷积，
大幅减少参数量和计算量。
"""

import torch
import torch.nn as nn


class DepthwiseConvBlock(nn.Module):
    """
    深度可分离卷积块

    结构：
        Input -> [1x1 Pointwise Conv] -> [3x3 Depthwise Conv] -> [BN] -> [ReLU6] -> Output

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小 (默认 3x3)
        stride: 步长
        expansion_ratio: 扩展比率 (默认 1，设为 6 类似 MobileNetV2 倒残差结构)
        use_residual: 是否使用残差连接
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 expansion_ratio=1, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        # 计算扩展后的中间通道数
        hidden_dim = int(in_channels * expansion_ratio)

        # 逐点卷积 (扩展通道)
        # 只有当扩展比率 > 1 时才实际增加通道，否则直接用恒等映射（不增加计算量）
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False), # 1x1 卷积升维
            nn.BatchNorm2d(hidden_dim),  # 批归一化
            nn.ReLU6(inplace=True)   # ReLU6 激活
        ) if expansion_ratio > 1 else nn.Identity()  # 不扩展通道，直接传递

        # 3x3 逐通道卷积 (空间卷积)
        # groups = hidden_dim（或 in_channels）意味着每个输入通道独立卷积，即深度卷积
        # 这一步只做空间上的特征提取，不改变通道数
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim if expansion_ratio > 1 else in_channels,
                     hidden_dim if expansion_ratio > 1 else in_channels,
                     kernel_size,
                     stride=stride, # 步长，控制下采样
                     padding=kernel_size//2,   # 保持空间尺寸（stride=1时）
                     groups=hidden_dim if expansion_ratio > 1 else in_channels,
                     bias=False),
            nn.BatchNorm2d(hidden_dim if expansion_ratio > 1 else in_channels),
            nn.ReLU6(inplace=True)
        )

        # 逐点卷积 (压缩通道)
        # 将中间通道数降为输出通道数 out_channels
        self.pw_compress = nn.Sequential(
            nn.Conv2d(hidden_dim if expansion_ratio > 1 else in_channels,
                     out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
            # 注意这里没有在序列中加 ReLU，因为残差相加后才会做最后的激活
        )

        # 残差连接 (仅当输入输出维度一致时)
        # 当输入输出尺寸和通道完全一致时，直接恒等映射
        # 否则使用 1x1 卷积调整维度，保证可以相加
        self.residual = nn.Identity()
        if use_residual and in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        elif in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x # 保留输入，用作残差连接
        # 第一步：通道扩展（如果 expansion_ratio > 1）
        out = self.pw_conv(x) if not isinstance(self.pw_conv, nn.Identity) else x
        # 第二步：逐通道空间卷积
        out = self.dw_conv(out)
        # 第三步：通道压缩
        out = self.pw_compress(out)
        # 第四步：残差相加（先调整 identity 到相同形状，再相加）
        if self.use_residual:
            out = out + self.residual(identity)
        # 最后统一过 ReLU6 激活
        out = nn.ReLU6(inplace=True)(out)
        return out


class Stem(nn.Module):
    """
    模型的 Stem 层，负责初始快速下采样和通道扩展。
    它把输入图像（如 3x256x256）迅速变为较小尺寸、较多通道的特征图，
    为后续阶段提供良好的起点。

    Args:
        in_channels: 输入图像的通道数，默认为 3（RGB）
        out_channels: 输出特征图的通道数
    """

    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        # Stem 结构：两次卷积，第一次 stride=2 下采样，第二次保持尺寸
        self.stem = nn.Sequential(
            # 第一层：3x3 卷积，步长 2，高宽减半，通道升到 out_channels//2
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU6(inplace=True),
            # 第二层：3x3 卷积，步长 1，保持尺寸，通道继续升到 out_channels
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        # 直接通过 stem 序列输出
        return self.stem(x)


class Stage(nn.Module):
    """
    模型的一个阶段（Stage），包含多个深度可分离卷积块。
    通常第一个块负责下采样，后续块保持分辨率并提取更深层特征。
    这里的 Stage 由参数 num_blocks 决定块的堆叠数量。

    Args:
        in_channels: 该阶段的输入通道数
        out_channels: 该阶段的输出通道数
        num_blocks: 堆叠的 DepthwiseConvBlock 数量
        stride: 第一个块的步长（通常设为 2 来实现下采样）
    """

    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super().__init__()
        blocks = []

        # 第一个块 (可能下采样)
        # stride 控制是否缩小特征图尺寸，并同时完成通道数的转换
        blocks.append(DepthwiseConvBlock(in_channels, out_channels, stride=stride))

        # 后续块 (无下采样)
        # 每个块接收和输出都是 out_channels，stride=1，不改变分辨率
        for _ in range(1, num_blocks):
            blocks.append(DepthwiseConvBlock(out_channels, out_channels, stride=1))

        # 使用 nn.Sequential 将块串联起来
        self.stage = nn.Sequential(*blocks)

    def forward(self, x):
        # 顺序通过所有块
        return self.stage(x)
