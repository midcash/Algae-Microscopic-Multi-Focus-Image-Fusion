"""
藻类感知决策先验模块

为多焦点源图像生成基于结构/清晰度的显式先验决策图，
用于辅助现有 DecisionMapFusion 更稳定地估计每个源图的融合权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlgaeAwareDecisionPrior(nn.Module):
    """
    生成每个源图的单通道结构/清晰度响应图，并归一化为先验决策图。

    当前支持两种模式：
    - gradient: Sobel 梯度幅值
    - gradient_contrast: Sobel 梯度幅值 + 局部对比度近似响应
    """

    def __init__(self, mode='gradient', eps=1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _to_gray(self, image):
        if image.size(1) == 1:
            return image
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _gradient_response(self, gray):
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.eps)

    def _contrast_response(self, gray):
        local_mean = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((gray - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        return torch.sqrt(local_var + self.eps)

    def _normalize_per_image(self, response):
        b = response.size(0)
        flat = response.view(b, -1)
        min_v = flat.min(dim=1)[0].view(b, 1, 1, 1)
        max_v = flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (response - min_v) / (max_v - min_v + self.eps)

    def forward(self, source_images, target_size=None):
        responses = []

        for src in source_images:
            gray = self._to_gray(src)
            response = self._gradient_response(gray)

            if self.mode == 'gradient_contrast':
                response = response + self._contrast_response(gray)
            elif self.mode != 'gradient':
                raise ValueError(f'Unsupported decision prior mode: {self.mode}')

            response = self._normalize_per_image(response)

            if target_size is not None and response.shape[2:] != target_size:
                response = F.interpolate(
                    response,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            responses.append(response)

        stacked = torch.cat(responses, dim=1)
        prior_weights = stacked / (stacked.sum(dim=1, keepdim=True) + self.eps)
        return prior_weights
