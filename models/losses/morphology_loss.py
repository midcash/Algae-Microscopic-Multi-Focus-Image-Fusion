"""
形态保持损失函数

面向藻类显微图像融合的第一版形态保持约束，重点增强：
1. 边界与细结构梯度保持
2. 高频形态细节保持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.boundary_loss import sobel_gradient


class MorphologyPreservingLoss(nn.Module):
    """
    形态保持损失 v1

    设计目标：
    - 让融合图像的梯度响应接近所有源图中的最强梯度响应
    - 让融合图像的高频细节接近所有源图中的最强高频响应

    Args:
        weight: 总损失权重
        gradient_weight: 梯度保持项权重
        laplacian_weight: 拉普拉斯细结构项权重
    """

    def __init__(self, weight=1.0, gradient_weight=1.0, laplacian_weight=0.5):
        super().__init__()
        self.weight = weight
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight

        laplacian_kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],
            dtype=torch.float32
        )
        self.register_buffer('laplacian_kernel', laplacian_kernel.unsqueeze(0).unsqueeze(0))

    def laplacian_response(self, img):
        """计算多通道图像的拉普拉斯响应"""
        kernel = self.laplacian_kernel.repeat(img.shape[1], 1, 1, 1)
        response = F.conv2d(img, kernel, padding=1, groups=img.shape[1])
        return torch.abs(response)

    def forward(self, fused, source_images):
        """
        Args:
            fused: 融合图像 (B, C, H, W)
            source_images: 源图像列表 List[Tensor]

        Returns:
            标量损失
        """
        # 梯度参考：保留所有源图中的最强边界/结构响应
        source_gradients = [sobel_gradient(src) for src in source_images]
        max_gradient = torch.max(torch.stack(source_gradients, dim=0), dim=0)[0]
        fused_gradient = sobel_gradient(fused)
        gradient_term = F.l1_loss(fused_gradient, max_gradient)

        # 高频细节参考：保留所有源图中的最强拉普拉斯响应
        source_laplacians = [self.laplacian_response(src) for src in source_images]
        max_laplacian = torch.max(torch.stack(source_laplacians, dim=0), dim=0)[0]
        fused_laplacian = self.laplacian_response(fused)
        laplacian_term = F.l1_loss(fused_laplacian, max_laplacian)

        loss = self.gradient_weight * gradient_term + self.laplacian_weight * laplacian_term
        return self.weight * loss
