"""
边界感知损失函数

借鉴 alpha-matte 思想，对焦边界区域进行建模，减少融合伪影。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_gradient(img):
    """
    使用 Sobel 算子计算图像梯度幅值

    Args:
        img: 输入图像 (B, C, H, W)

    Returns:
        梯度幅值图 (B, C, H, W)
    """
    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32)

    if img.is_cuda:
        sobel_x = sobel_x.cuda(img.get_device())
        sobel_y = sobel_y.cuda(img.get_device())

    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

    # 对每个通道分别计算梯度
    grad_x = F.conv2d(img, sobel_x.expand(img.shape[1], -1, -1, -1),
                      padding=1, groups=img.shape[1])
    grad_y = F.conv2d(img, sobel_y.expand(img.shape[1], -1, -1, -1),
                      padding=1, groups=img.shape[1])

    # 梯度幅值
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    return magnitude


class BoundaryLoss(nn.Module):
    """
    边界感知损失

    借鉴 alpha-matte 思想，对焦边界区域进行特殊处理。

    Args:
        weight: 损失权重

    Algorithm:
        1. 使用 Sobel 算子检测各源图的梯度幅值
        2. 选择最大梯度响应作为边界参考图
        3. 计算融合图像梯度与边界参考图的 L1 距离

    Equation:
        L_boundary = ||∇fused - max(∇source_i)||_1

    为什么有效:
        - 多聚焦图像中，每个源图在不同区域有不同的聚焦程度
        - 聚焦区域的梯度响应更强
        - 通过选择最大梯度作为参考，可以获取所有源图中的清晰边界
        - 约束融合图像的梯度接近最大梯度，可以保证边界清晰
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, fused, source_images):
        """
        Args:
            fused: 融合图像 (B, C, H, W)
            source_images: 源图像列表 List[Tensor]

        Returns:
            边界损失 (标量)
        """
        # 计算各源图的梯度幅值
        gradients = [sobel_gradient(src) for src in source_images]

        # 选择最大梯度作为边界参考 (alpha-matte 思想)
        max_gradient = torch.max(torch.stack(gradients), dim=0)[0]

        # 计算融合图像的梯度
        fused_gradient = sobel_gradient(fused)

        # L1 边界损失
        loss = F.l1_loss(fused_gradient, max_gradient)

        return self.weight * loss


class BoundaryLossWithMask(nn.Module):
    """
    带掩码的边界感知损失

    仅对边界区域计算损失，减少平滑区域的干扰。

    Args:
        weight: 损失权重
        threshold: 边界检测阈值
        dilation: 边界膨胀次数 (扩大边界区域)

    Algorithm:
        1. 计算最大梯度图
        2. 阈值化得到边界掩码
        3. 形态学膨胀扩大边界区域
        4. 仅在边界区域计算损失
    """

    def __init__(self, weight=1.0, threshold=0.1, dilation=3):
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.dilation = dilation

    def forward(self, fused, source_images):
        """
        Args:
            fused: 融合图像 (B, C, H, W)
            source_images: 源图像列表 List[Tensor]

        Returns:
            带掩码的边界损失 (标量)
        """
        # 计算各源图的梯度幅值
        gradients = [sobel_gradient(src) for src in source_images]
        max_gradient = torch.max(torch.stack(gradients), dim=0)[0]
        fused_gradient = sobel_gradient(fused)

        # 生成边界掩码
        # 归一化梯度
        max_gradient_norm = max_gradient / (max_gradient.max() + 1e-8)

        # 阈值化
        mask = (max_gradient_norm > self.threshold).float()

        # 形态学膨胀 (扩大边界区域)
        for _ in range(self.dilation):
            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

        # 仅在边界区域计算损失
        loss = F.l1_loss(fused_gradient * mask, max_gradient * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-8)  # 归一化

        return self.weight * loss
