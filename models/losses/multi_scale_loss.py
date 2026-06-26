"""
多尺度一致性损失函数

在不同尺度下约束融合质量，确保多尺度藻类都能良好融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.fusion_loss import ssim_loss, gradient_loss


class MultiScaleLoss(nn.Module):
    """
    多尺度一致性损失

    通过在多个尺度下计算融合损失，确保不同大小的藻类目标都能良好融合。

    Args:
        weight: 损失权重
        num_scales: 尺度数量
        scale_factor: 尺度缩放因子 (默认 0.5，即每层缩小一半)

    Equation:
        L_multi_scale = Σ(scale_factor^s * L_fusion(downsample^s(fused), downsample^s(source_i)))

    为什么有效:
        - 藻类尺寸差异大 (2-4μm 到 100μm)
        - 单一尺度下难以同时捕捉大小目标
        - 多尺度损失可以在不同分辨率下约束融合质量
        - 大尺度捕捉小藻类细节，小尺度捕捉大藻类结构
    """

    def __init__(self, weight=1.0, num_scales=3, scale_factor=0.5):
        super().__init__()
        self.weight = weight
        self.num_scales = num_scales
        self.scale_factor = scale_factor

    def forward(self, fused, source_images):
        """
        Args:
            fused: 融合图像 (B, C, H, W)
            source_images: 源图像列表 List[Tensor]

        Returns:
            多尺度损失 (标量)
        """
        loss = 0

        for s in range(self.num_scales):
            scale = self.scale_factor ** s

            if scale < 1.0:
                # 下采样
                fused_s = F.interpolate(fused, scale_factor=scale, mode='bilinear', align_corners=False)
                sources_s = [
                    F.interpolate(src, scale_factor=scale, mode='bilinear', align_corners=False)
                    for src in source_images
                ]
            else:
                fused_s = fused
                sources_s = source_images

            # 计算当前尺度的融合损失
            scale_loss = 0
            for src in sources_s:
                # SSIM 损失
                ssim = ssim_loss(fused_s, src)
                # 梯度损失
                grad = gradient_loss(fused_s, src)
                scale_loss += ssim + 0.5 * grad

            scale_loss /= len(source_images)

            # 加权累加 (大尺度权重更高)
            loss += scale * scale_loss

        return self.weight * loss


class PyramidLoss(nn.Module):
    """
    金字塔损失

    类似拉普拉斯金字塔，在多个尺度下计算残差损失。

    Args:
        weight: 损失权重
        num_levels: 金字塔层数

    Algorithm:
        1. 构建融合图像和源图像的高斯金字塔
        2. 计算相邻层的拉普拉斯残差
        3. 在每层比较融合图像与源图像的残差
    """

    def __init__(self, weight=1.0, num_levels=4):
        super().__init__()
        self.weight = weight
        self.num_levels = num_levels

    def _gaussian_pyramid(self, img, num_levels):
        """构建高斯金字塔"""
        pyramid = [img]
        for _ in range(num_levels - 1):
            img = F.avg_pool2d(img, kernel_size=2, stride=2)
            pyramid.append(img)
        return pyramid

    def _laplacian_pyramid(self, pyramid):
        """构建拉普拉斯金字塔"""
        laplacian = []
        for i in range(len(pyramid) - 1):
            upsampled = F.interpolate(pyramid[i + 1], size=pyramid[i].shape[2:], mode='bilinear', align_corners=False)
            laplacian.append(pyramid[i] - upsampled)
        laplacian.append(pyramid[-1])  # 最底层
        return laplacian

    def forward(self, fused, source_images):
        """
        Args:
            fused: 融合图像 (B, C, H, W)
            source_images: 源图像列表 List[Tensor]

        Returns:
            金字塔损失 (标量)
        """
        # 构建融合图像的金字塔
        fused_gaussian = self._gaussian_pyramid(fused, self.num_levels)
        fused_laplacian = self._laplacian_pyramid(fused_gaussian)

        loss = 0

        for src in source_images:
            # 构建源图像的金字塔
            src_gaussian = self._gaussian_pyramid(src, self.num_levels)
            src_laplacian = self._laplacian_pyramid(src_gaussian)

            # 比较每层拉普拉斯残差
            for fused_lap, src_lap in zip(fused_laplacian, src_laplacian):
                loss += F.l1_loss(fused_lap, src_lap)

        loss /= len(source_images)

        return self.weight * loss
