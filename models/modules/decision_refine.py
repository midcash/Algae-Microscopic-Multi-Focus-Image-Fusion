"""
R22: Learnable DecisionRefine — 轻量决策精炼模块

核心思想（论文叙事）：
  硬选择 + 结构引导精炼 = 保持边缘锐利 + 消除碎片化

  在 GumbelDecisionFusion 输出 logits 后，
  用一个小型可学习模块对决策图进行空间精炼：
  - 输入：raw logits (B, N, H, W) + decoder features (B, C, H, W)
  - 精炼：3x3 depthwise conv（局部平滑）+ 残差连接（保持原始决策）
  - 输出：refined logits → 再硬选择

  这不是否定硬选择，而是"精炼硬选择"：
  模型先做硬决策，再根据局部上下文修正明显错误的孤立像素。

参数量: ~200（极小，不影响轻量化）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionRefine(nn.Module):
    """
    轻量决策精炼模块

    在 logit 空间对决策进行局部平滑精炼，
    用 decoder 特征作为 guidance 判断哪些像素需要修正。
    """

    def __init__(self, num_classes=5, guide_channels=8, refine_strength=0.5):
        super().__init__()
        self.refine_strength = refine_strength

        # 超轻量：depthwise conv 做局部平滑
        self.spatial_smooth = nn.Conv2d(
            num_classes, num_classes, 3, padding=1, groups=num_classes, bias=False
        )
        # 初始化为简单的均值滤波
        nn.init.constant_(self.spatial_smooth.weight, 1.0 / 9.0)

        # Guidance gate：decoder特征 → 判断哪些区域需要精炼
        self.gate = nn.Sequential(
            nn.Conv2d(guide_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, logits, decoder_features):
        """
        Args:
            logits: (B, N, H, W) DecisionNet 原始 logits
            decoder_features: (B, C, H, W) 解码器特征
        Returns:
            refined_logits: (B, N, H, W) 精炼后的 logits
        """
        # 1. 局部平滑
        smoothed = self.spatial_smooth(logits)  # (B, N, H, W)

        # 2. 门控：平坦区域多平滑，边缘区域少平滑
        gate = self.gate(decoder_features)  # (B, 1, H, W)

        # 3. 残差融合：平滑的 logits + 原始 logits
        # gate=1 → 多平滑（平坦区），gate=0 → 保持原样（边缘区）
        refined = logits + self.refine_strength * gate * (smoothed - logits)

        return refined
