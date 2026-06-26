"""
融合损失函数

基于结构相似性 (SSIM) 和梯度一致性的无监督融合损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(size=11, sigma=1.5):
    """
    生成 2D 高斯核

    Args:
        size: 核大小
        sigma: 标准差

    Returns:
        2D 高斯核 Tensor (1, 1, size, size)
    """
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    kernel = g.outer(g)
    return kernel.unsqueeze(0).unsqueeze(0)


def ssim_loss(img1, img2, kernel_size=11, sigma=1.5):
    """
    计算 SSIM 损失

    Args:
        img1: 图像 1 (B, C, H, W)
        img2: 图像 2 (B, C, H, W)
        kernel_size: 高斯核大小
        sigma: 高斯核标准差

    Returns:
        SSIM 损失 (标量)

    Equation:
        SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
        Loss = 1 - SSIM
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel = gaussian_kernel(kernel_size, sigma).to(img1.device)
    # kernel shape: (1, 1, size, size) -> (C, 1, size, size) for groups=C
    kernel = kernel.repeat(img1.shape[1], 1, 1, 1)

    # 计算局部均值
    mu1 = F.conv2d(img1, kernel, padding=kernel_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=kernel_size//2, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算局部方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_mu2

    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()


def gradient_loss(img1, img2):
    """
    计算梯度损失

    Args:
        img1: 图像 1 (B, C, H, W)
        img2: 图像 2 (B, C, H, W)

    Returns:
        梯度损失 (标量)
    """
    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).to(img1.device)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).to(img1.device)

    # 复制卷积核到每个通道 (用于 groups=C 的深度卷积)
    sobel_x = sobel_x.repeat(img1.shape[1], 1, 1, 1)
    sobel_y = sobel_y.repeat(img1.shape[1], 1, 1, 1)

    # 计算梯度
    grad1_x = F.conv2d(img1, sobel_x, padding=1, groups=img1.shape[1])
    grad1_y = F.conv2d(img1, sobel_y, padding=1, groups=img1.shape[1])
    grad2_x = F.conv2d(img2, sobel_x, padding=1, groups=img2.shape[1])
    grad2_y = F.conv2d(img2, sobel_y, padding=1, groups=img2.shape[1])

    # 梯度幅值
    grad1 = torch.sqrt(grad1_x ** 2 + grad1_y ** 2 + 1e-8)
    grad2 = torch.sqrt(grad2_x ** 2 + grad2_y ** 2 + 1e-8)

    return F.l1_loss(grad1, grad2)


class FusionLoss(nn.Module):
    """
    融合损失函数

    结合 SSIM 损失和梯度损失，用于无监督多聚焦图像融合。

    Args:
        ssim_weight: SSIM 损失权重
        grad_weight: 梯度损失权重
        ssim_kernel_size: SSIM 高斯核大小
        ssim_sigma: SSIM 高斯核标准差

    Equation:
        L_fusion = Σ(SSIM(fused, source_i) * max(∇fused, ∇source_i))
    """

    def __init__(self, ssim_weight=1.0, grad_weight=0.5,
                 ssim_kernel_size=11, ssim_sigma=1.5):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.ssim_kernel_size = ssim_kernel_size
        self.ssim_sigma = ssim_sigma

    def forward(self, fused, source_images):
        """
        Args:
            fused: 融合图像 (B, C, H, W)
            source_images: 源图像列表 List[Tensor], 每个 (B, C, H, W)

        Returns:
            融合损失 (标量)
        """
        loss = 0

        for src in source_images:
            # SSIM 损失
            ssim = ssim_loss(fused, src, self.ssim_kernel_size, self.ssim_sigma)

            # 梯度损失
            grad = gradient_loss(fused, src)

            loss += self.ssim_weight * ssim + self.grad_weight * grad

        return loss / len(source_images)
