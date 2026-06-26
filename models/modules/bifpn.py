"""
BiFPN 模块 - 双向特征金字塔网络

Bidirectional Feature Pyramid Network, 来自 EfficientDet 的高效多尺度特征融合模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNLayer(nn.Module):
    """
    单层 BiFPN 模块

    完成一次完整的“自上而下 + 自底向上”的特征融合。
    每一层内包含多组可学习的融合权重，以及用于后续处理的卷积。

    Args:
        channels: 每个特征层级的通道数（所有层级统一成相同通道数）
        num_levels: 特征金字塔的层数，例如 4 表示 P2, P3, P4, P5

    Structure:
        1. 自顶向下路径：高层特征上采样后与低层特征融合
        2. 自底向上路径：低层特征下采样后与高层特征融合
        3. 每个融合点拥有独立可学习权重
    """

    def __init__(self, channels, num_levels=4, eps=1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.channels = channels
        self.eps = eps

        # ------------------------------------------------------------
        # 可学习的融合权重
        # ------------------------------------------------------------
        # 每个融合点需要两个权重：一个给本层原始特征，一个给传来的另一层特征。
        # 自上而下路径有 (num_levels - 1) 个融合点，
        # 自底向上路径也有 (num_levels - 1) 个融合点。
        # 每个权重向量形状为 (2,)，初始化为 1（等权重开始）。
        self.up_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(num_levels - 1)
        ])
        self.down_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(num_levels - 1)
        ])

        # ------------------------------------------------------------
        # 特征转换卷积（每个层级一个）
        # ------------------------------------------------------------
        # 融合后的特征可能有些混叠，用 3x3 卷积做一次平滑/精炼。
        self.feature_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True)
            )
            for _ in range(num_levels)
        ])

        # ------------------------------------------------------------
        # 上采样模块（用于将深层小图放大到浅层大图尺寸）
        # ------------------------------------------------------------
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(num_levels - 1)
        ])

        # ------------------------------------------------------------
        # 下采样模块（用于将浅层大图缩小到深层小图尺寸）
        # ------------------------------------------------------------
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True)
            )
            for _ in range(num_levels - 1)
        ])

    def _normalized_weights(self, weights):
        """
        权重归一化：先经过 ReLU 保证非负，然后除以总和变成概率分布。
        这样每个融合点的两个输入权重加起来等于 1，可以插值混合。
        """
        weights = F.relu(weights)
        return weights / (weights.sum() + self.eps)

    def _fuse_pair(self, left, right, weights):
        """
        使用归一化后的权重对两个特征图进行加权融合。
        left: 本层原始特征（或上一轮融合后的特征）
        right: 从相邻层传过来的特征（上采样或下采样后）
        weights: 长度为 2 的可学习权重向量
        """
        norm_w = self._normalized_weights(weights)
        # 加权和：w0 * left + w1 * right
        return norm_w[0] * left + norm_w[1] * right

    def forward(self, features):
        """
        输入：features —— 一个列表，包含从高分辨率到低分辨率的多尺度特征。
              例如 [P2, P3, P4, P5]，索引 0 分辨率最高，索引 -1 分辨率最低。
        输出：融合后的同形状多尺度特征列表。
        """
        features = list(features)  # 复制一份，避免修改原列表

        # ============================================================
        # 第一阶段：自上而下路径 (Top-down Pathway)
        # 从最低分辨率（P5）开始，逐步上采样并融合到更高分辨率层
        # 方向：P5 → P4 → P3 → P2
        # ============================================================
        td_features = list(features)
        for i in range(self.num_levels - 1, 0, -1):  # i = 3,2,1 (对应 P5→P4, P4→P3, P3→P2)
            # 将当前低分辨率特征图上采样到上一层的尺寸
            upsampled = self.upsamples[i - 1](td_features[i])
            # 与上一层原始特征加权融合
            td_features[i - 1] = self._fuse_pair(td_features[i - 1], upsampled, self.up_weights[i - 1])

        # ============================================================
        # 第二阶段：自底向上路径 (Bottom-up Pathway)
        # 从最高分辨率（P2）开始，逐步下采样并融合到更低分辨率层
        # 方向：P2 → P3 → P4 → P5
        # ============================================================
        out_features = [td_features[0]] # P2 作为起点
        for i in range(self.num_levels - 1): # i = 0,1,2
            # 将当前较高分辨率特征图下采样到下一层的尺寸
            downsampled = self.downsamples[i](out_features[i])
            # 与下一层自上而下路径产生的特征加权融合
            fused = self._fuse_pair(td_features[i + 1], downsampled, self.down_weights[i])
            out_features.append(fused)

        # ============================================================
        # 最后：对每个层级的融合特征做一次 3x3 卷积精炼
        # ============================================================
        out_features = [self.feature_convs[i](f) for i, f in enumerate(out_features)]
        return out_features


class BiFPN(nn.Module):
    """
    多层 BiFPN 模块

    将多个 BiFPNLayer 堆叠起来，进一步增强多尺度特征交互。
    同时负责输入特征的通道统一和最终的特征精炼。

    Args:
        in_channels_list: 各层级输入通道数列表
        out_channels: 输出通道数
        num_levels: 特征金字塔层级数
        num_layers: BiFPN 层数

    Example:
        # 输入：[C, 2C, 4C, 8C] 对应 [P2, P3, P4, P5]
        bifpn = BiFPN(
            in_channels_list=[64, 128, 256, 512],
            out_channels=128,
            num_levels=4,
            num_layers=2
        )
    """

    def __init__(self, in_channels_list, out_channels, num_levels=4, num_layers=2):
        super().__init__()
        self.num_levels = num_levels
        self.num_layers = num_layers

        # ------------------------------------------------------------
        # 输入投影：将各层级可能不同的通道数统一投影到 out_channels
        # ------------------------------------------------------------
        self.input_projs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1, bias=False) if in_c != out_channels else nn.Identity()
            for in_c in in_channels_list
        ])

        # ------------------------------------------------------------
        # BiFPN 核心层，每一层都做一次完整的双向融合
        # ------------------------------------------------------------
        self.layers = nn.ModuleList([
            BiFPNLayer(out_channels, num_levels)
            for _ in range(num_layers)
        ])

        # ------------------------------------------------------------
        # 输出投影：一个共享的 3x3 卷积，进一步平滑融合结果
        # ------------
        self.output_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, features):
        """
        输入：features —— 列表，包含各层级特征，通道数可能各不相同
        输出：融合后的统一通道多尺度特征列表
        """
        # 第一步：通道数统一
        features = [proj(f) for proj, f in zip(self.input_projs, features)]

        # 第二步：逐层经过 BiFPNLayer
        for layer in self.layers:
            features = layer(features)

        # 第三步：输出精炼
        features = [self.output_proj(f) for f in features]

        return features


class FeaturePyramid(nn.Module):
    """
    特征金字塔构建器

    从编码器输出的多阶段特征（例如 [C1, C2, C3, C4]）出发，
    额外下采样生成更高层级的语义特征（如 P5, P6），
    然后送入 BiFPN 进行多尺度融合。

    Args:
        encoder_channels: 编码器各阶段输出通道数
        out_channels: BiFPN 输出通道数
    """

    def __init__(self, encoder_channels, out_channels):
        super().__init__()
        # 假设 encoder_channels = [32, 64, 128, 256]
        # 用于从编码器最深层特征再下采样两次，得到 P5, P6
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_channels[-1], encoder_channels[-1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(encoder_channels[-1]),
                nn.ReLU6(inplace=True)
            )
            for _ in range(2)  # 额外下采样 2 层
        ])

        # 构建 BiFPN，输入通道列表为原始四层 + 新增两层
        self.bifpn = BiFPN(
            in_channels_list=list(encoder_channels) + [encoder_channels[-1]] * 2,
            out_channels=out_channels,
            num_levels=6, # 共 6 层：原 4 层 + 新增 2 层
            num_layers=2
        )

    def forward(self, stage_outputs):
        """
        Args:
            stage_outputs: List[Tensor], 编码器各阶段输出

        Returns:
            List[Tensor], BiFPN 输出的多尺度特征
        """
        features = list(stage_outputs)

        # 从最深特征下采样两次，堆叠到金字塔上
        x = features[-1]
        for down in self.downsamples:
            x = down(x)
            features.append(x)

        # 送入 BiFPN
        return self.bifpn(features)
