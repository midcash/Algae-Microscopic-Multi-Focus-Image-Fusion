"""
R12 DecisionNet 改进版
- 输入特征改为多尺度梯度幅值差异（直接编码清晰度）
- 新增梯度对比损失：选中的源图梯度应大于其他源图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gradient_magnitude(img):
    """
   计算多通道图像的梯度幅值，用于衡量每个像素的“边缘强度”或“清晰度”。

   原理：
   使用 Sobel 算子分别计算水平方向 (x) 和垂直方向 (y) 的梯度，
   然后合成梯度幅值 = sqrt(gx² + gy²)。
   对彩色图像，先对各通道独立计算梯度，然后取平均，得到单通道的梯度图。

   Args:
       img: (B, C, H, W) 输入图像张量

   Returns:
       grad: (B, 1, H, W) 梯度幅值图，每个像素值越大表示边缘越明显、越清晰
   """
    # 定义 Sobel 卷积核（水平方向和垂直方向）
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(img.device)

    # 将卷积核变形为 (out_channels, in_channels, kh, kw)，并用 groups 实现逐通道卷积
    # 这里每个通道单独做 Sobel，所以 groups=img.shape[1]
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(img.shape[1], 1, 1, 1)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(img.shape[1], 1, 1, 1)

    # 计算水平和垂直梯度（逐通道，padding=1 保持尺寸）
    gx = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    gy = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])

    # 梯度幅值 = sqrt(gx² + gy²)，加 1e-8 防止根号下为 0
    grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)

    # 在通道维度取平均，得到单通道的梯度图 (B, 1, H, W)
    return grad.mean(dim=1, keepdim=True)  # (B, 1, H, W)


class DecisionNet(nn.Module):
    """
    决策网络：根据多源图像的多尺度梯度差异，为每个像素预测“应该选哪张源图”。

    设计思路：
    1. 清晰区域梯度幅值大，模糊区域梯度小。
       通过计算每对源图的梯度幅值差，网络可以直接感知“哪张图这里更清晰”。
    2. 多尺度：在不同分辨率下比较梯度，既能捕捉大范围模糊区域，也能精细定位边缘。
    3. 可选解码器特征：外部解码器提供的上下文特征（如整体结构）也可以辅助决策，
       通过投影后与梯度差异特征相加，实现特征融合。
    4. 最终输出 N 个通道的 logits，代表每个像素属于各源图的分数。

    Args:
        num_source_images: 源图像数量（默认 5）
        num_scales: 用于计算梯度差异的尺度数（默认 3，即原图、1/2、1/4）
        base_channels: 内部特征通道数
    """
    def __init__(self, num_source_images=5, num_scales=3, base_channels=32):
        super().__init__()
        self.num_source_images = num_source_images
        self.num_scales = num_scales

        # 计算输入特征图的通道数
        # 每对源图 (i,j) 产生一个梯度差异图，总共有 C(N,2) 对
        num_pairs = num_source_images * (num_source_images - 1) // 2

        # 每个尺度都有 num_pairs 个差异图，把它们在通道上拼接
        in_channels = num_pairs * num_scales

        # 梯度差异处理分支
        self.grad_ops = nn.Sequential(
            # 1. 普通 3x3 卷积，将输入投影到 base_channels
            nn.Conv2d(in_channels, base_channels, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # 2. 深度可分离卷积（等价于 MobileNet 风格）进一步提取特征
            #    先逐通道 3x3 卷积
            nn.Conv2d(base_channels, base_channels, 3, padding=1,
                      groups=base_channels, bias=False),
            #    再接 1x1 逐点卷积混合通道
            nn.Conv2d(base_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        # 输出层：3x3 卷积，输出 N 个通道的 logits
        self.out_conv = nn.Conv2d(base_channels, num_source_images, 3, padding=1, bias=True)

    def _compute_gradient_diff_feats(self, source_images):
        """
        计算多尺度梯度差异特征。

        过程：
        1. 对每个尺度，将源图缩放到该尺度（原图、1/2 尺寸、1/4 尺寸）。
        2. 计算每张图的梯度幅值图。
        3. 对所有源图两两配对，计算梯度幅值的绝对差异 |grad_i - grad_j|。
        4. 将该尺度下所有配对的差异图在通道上拼接。
        5. 把所有尺度的拼接结果再次在通道上拼接，形成最终的特征张量。

        这样产生的特征编码了“在哪个尺度、哪张图更清晰”的信息，
        并且空间尺寸统一为原始高宽 H, W。

        Returns:
            (B, num_pairs * num_scales, H, W)
        """
        N = len(source_images)
        B, _, H, W = source_images[0].shape
        scales = [1.0, 0.5, 0.25]  # 三个尺度：原图、1/2、1/4
        num_pairs = N * (N - 1) // 2

        all_diff = [] # 存储各尺度的差异特征

        for scale in scales:
            if scale < 1.0:
                # 缩小图像到指定尺寸
                sH, sW = int(H * scale), int(W * scale)
                scaled = [F.interpolate(src, size=(sH, sW), mode='bilinear', align_corners=False)
                          for src in source_images]
            else:
                scaled = source_images
                sH, sW = H, W

            # 对各源图计算梯度幅值
            grads = [_gradient_magnitude(src) for src in scaled]  # 每个是 (B, 1, sH, sW)

            # 计算所有两两配对的梯度差异
            pair_diffs = []
            for i in range(N):
                for j in range(i + 1, N):
                    diff = torch.abs(grads[i] - grads[j])  # (B, 1, sH, sW)
                    pair_diffs.append(diff)

            if pair_diffs:
                # 将所有配对的差异图在通道上拼接，得到 (B, num_pairs, sH, sW)
                scale_feat = torch.cat(pair_diffs, dim=1)  # (B, num_pairs, sH, sW)
                # 如果不是原图尺寸，则上采样回原图 H, W
                if scale < 1.0:
                    scale_feat = F.interpolate(scale_feat, size=(H, W), mode='bilinear', align_corners=False)
                all_diff.append(scale_feat)

            # 将所有尺度的特征在通道上拼接，最终通道数为 num_pairs * num_scales
        return torch.cat(all_diff, dim=1)

    def forward(self, source_images, decoder_feat=None, decoder_proj=None, decoder_gate=None):
        """
       前向传播。

       Args:
           source_images: List[Tensor] 长度为 N，每个张量形状 (B, 3, H, W)
           decoder_feat: (B, C_dec, H, W) 解码器特征（可选），用于注入全局上下文
           decoder_proj: 一个 nn.Module，将 decoder_feat 投影到 base_channels（可选）
       Returns:
           logits: (B, N, H, W) 每个像素在 N 张源图上的分数
       """
        # 1. 计算梯度差异特征
        diff_feats = self._compute_gradient_diff_feats(source_images)
        # 2. 通过梯度处理分支提取特征
        feat = self.grad_ops(diff_feats)  # (B, base_channels, H, W)

        # R15/V5-min: 融合解码器特征
        # 3. 如果提供了解码器特征，则投影并融合
        if decoder_feat is not None and decoder_proj is not None:
            # 投影到 base_channels
            dec_feat = decoder_proj(decoder_feat)  # (B, base_channels, H, W)

            # 确保空间尺寸匹配 (通常已经一致）
            if dec_feat.shape[2:] != feat.shape[2:]:
                dec_feat = F.interpolate(dec_feat, size=feat.shape[2:],
                                         mode='bilinear', align_corners=False)

            # V5-min: 用 decoder 上下文生成 gate，显式决定更信梯度还是更信语义
            if decoder_gate is not None:
                gate = decoder_gate(decoder_feat)  # (B,1,H,W) or (B,C,H,W)
                if gate.shape[2:] != feat.shape[2:]:
                    gate = F.interpolate(gate, size=feat.shape[2:],
                                         mode='bilinear', align_corners=False)
                if gate.shape[1] == 1 and feat.shape[1] != 1:
                    gate = gate.expand(-1, feat.shape[1], -1, -1)
                feat = gate * feat + (1.0 - gate) * dec_feat
            else:
                # 回退：简单相加融合
                feat = feat + dec_feat

        # 4. 输出 logits
        logits = self.out_conv(feat)
        return logits


def gumbel_softmax_hard(logits, tau=0.67, dim=1):
    """
    Gumbel-Softmax 可微分硬选择（使用 Straight-Through Estimator, STE）

    原理：
    要让网络输出离散的 one-hot 选择（比如每个像素只选一张源图），
    但 argmax 操作不可导，无法反向传播。
    Gumbel-Softmax 技巧：在 logits 上加 Gumbel 噪声，然后 softmax，
    前向传播时用硬 one-hot，反向传播时用软 softmax 梯度近似。

    Args:
        logits: (B, N, H, W) 原始分数
        tau: 温度，控制 softmax 的尖锐度（越小越接近 one-hot）
        dim: 进行 softmax 的维度（默认 1，即通道维）

    Returns:
        hard_onehot_with_grad: (B, N, H, W) 前向是 one-hot，反向有梯度
    """
    B, N, H, W = logits.shape

    # 1. 生成 Gumbel 噪声：g = -log(-log(u)), u ~ Uniform(0,1)
    uniform = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-10)
    gumbel = -torch.log(-torch.log(uniform))

    # 2. 加噪声并缩放（除以温度）
    noisy_logits = (logits + gumbel) / tau

    # 3. softmax 得到软概率
    soft_probs = F.softmax(noisy_logits, dim=dim)

    # 4. 硬选择：argmax 得到最大值索引，转为 one-hot
    hard_indices = soft_probs.argmax(dim=dim, keepdim=True)  # (B, 1, H, W)
    hard_onehot = torch.zeros_like(soft_probs).scatter_(dim, hard_indices, 1.0)

    # 5. Straight-Through: 前向用硬 one-hot，反向梯度通过 soft_probs
    #    公式： output = hard_onehot.detach() + soft_probs - soft_probs.detach()
    #    这样 forward 时值是 hard_onehot，backward 时梯度等于 soft_probs 的梯度
    return hard_onehot.detach() + soft_probs - soft_probs.detach()


def select_and_fuse(source_images, decision_map):
    """
    根据决策图（硬选择权重）将多张源图融合为一张图像。

    Args:
        source_images: List[Tensor]，每个 (B, 3, H, W)
        decision_map: (B, N, 1, H, W) 硬选择权重，每个像素只有一个通道为 1，其余为 0
    Returns:
        fused: (B, 3, H, W) 融合图像
    """
    N = len(source_images)
    B, _, H, W = source_images[0].shape
    fused = torch.zeros_like(source_images[0])
    for i in range(N):
        # 取出第 i 张源图的权重，形状 (B, 1, 1, H, W)
        weight = decision_map[:, i:i+1]  # 保留维度，方便广播
        # 压缩掉通道维度，变成 (B, 1, H, W)，再扩展到 3 通道 (B, 3, H, W)
        weight = weight.squeeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)
        # 加权累加：由于 decision_map 是 one-hot，实际上就是复制选中的像素
        fused += weight * source_images[i]
    return fused


def bilateral_refine_decision(decision_map, guide_img, kernel_size=5,
                               sigma_spatial=2.0, sigma_color=0.1):
    """
    R22 (auto-experiment): 双边滤波精炼硬决策图

    核心思想：硬决策 + 结构引导的精炼，不改变硬选择范式。
    - 在同质区域（颜色相近的相邻像素）：平滑决策，消除碎片
    - 在结构边界（颜色突变的相邻像素）：保持硬决策，不跨边界混合

    这保留了论文"硬选择"的核心创新，同时解决低置信区碎片化问题。

    Args:
        decision_map: (B, N, 1, H, W) 硬决策 one-hot
        guide_img: (B, 3, H, W) 引导图像（提供结构信息）
        kernel_size: 滤波窗口大小
        sigma_spatial: 空间 Sigma
        sigma_color: 颜色 Sigma

    Returns:
        refined_map: (B, N, 1, H, W) 精炼决策（仍保持硬选择特性）
    """
    B, N, _, H, W = decision_map.shape
    device = decision_map.device
    k = kernel_size
    pad = k // 2

    # 引导图转灰度
    guide_gray = guide_img.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # 构建空间权重核
    coords = torch.arange(k, dtype=torch.float32, device=device) - pad
    gy, gx = torch.meshgrid(coords, coords, indexing='ij')
    spatial_kernel = torch.exp(-(gy**2 + gx**2) / (2 * sigma_spatial**2))
    spatial_kernel = spatial_kernel.view(1, 1, k, k)  # (1, 1, k, k)

    # 对每个源通道做引导滤波
    # 简化实现：用 guided filter 思路，先对 decision 做 box filter 近似
    dec = decision_map.squeeze(2)  # (B, N, H, W)

    refined_list = []
    for i in range(N):
        ch = dec[:, i:i+1, :, :]  # (B, 1, H, W)

        # 方法：在每个像素处，收集邻域像素的决策值，
        # 用空间距离 + 颜色距离加权平均
        # 用 unfold 提取 patches（不使用padding以避免尺寸不匹配）
        ch_pad = F.pad(ch, (pad, pad, pad, pad), mode='reflect')
        guide_pad = F.pad(guide_gray, (pad, pad, pad, pad), mode='reflect')

        # unfold: (B, C*k*k, H*W)
        ch_patches = F.unfold(ch_pad, kernel_size=k)  # (B, k*k, H*W)
        guide_patches = F.unfold(guide_pad, kernel_size=k)  # (B, k*k, H*W)

        # 中心像素的引导值
        center_idx = k * k // 2
        center_guide = guide_patches[:, center_idx:center_idx+1, :]  # (B, 1, H*W)

        # 颜色权重
        color_diff = (guide_patches - center_guide).abs()
        color_w = torch.exp(-color_diff / (2 * sigma_color**2 + 1e-8))  # (B, k*k, H*W)

        # 空间权重
        spatial_w = spatial_kernel.view(1, k*k, 1).to(device)  # (1, k*k, 1)

        # 组合权重
        total_w = color_w * spatial_w  # (B, k*k, H*W)

        # 加权平均
        weighted = (ch_patches * total_w).sum(dim=1, keepdim=True)  # (B, 1, H*W)
        norm = total_w.sum(dim=1, keepdim=True) + 1e-8
        refined_ch = (weighted / norm).view(B, 1, H, W)  # (B, 1, H, W)
        refined_list.append(refined_ch)

    refined = torch.cat(refined_list, dim=1)  # (B, N, H, W)

    # 重新硬化：取最大值，保持硬选择范式
    # 这样在平坦区域会平滑决策，但在边界处仍保持硬选择
    hard_idx = refined.argmax(dim=1, keepdim=True)  # (B, 1, H, W)
    hard_refined = torch.zeros_like(refined).scatter_(1, hard_idx, 1.0)  # (B, N, H, W)
    hard_refined = hard_refined.unsqueeze(2)  # (B, N, 1, H, W)

    return hard_refined


def bilateral_select_and_fuse(source_images, decision_map, guide_img=None,
                               kernel_size=5, sigma_spatial=2.0, sigma_color=0.1):
    """
    双边精炼 + 硬选择融合（推理使用）
    保持论文"硬选择"范式：先精炼决策图，再硬选择融合
    """
    if guide_img is None:
        guide_img = torch.stack(source_images, dim=0).mean(dim=0)

    refined_map = bilateral_refine_decision(
        decision_map, guide_img,
        kernel_size=kernel_size,
        sigma_spatial=sigma_spatial,
        sigma_color=sigma_color,
    )
    # 硬选择融合
    return select_and_fuse(source_images, refined_map)


def total_variation_loss(decision_map):
    """
    全变分损失（Total Variation Loss）—— 衡量决策图的空间平滑度。

    我们希望决策图不要有过多噪声和小块突变，边缘应该连续。
    通过计算相邻像素在决策概率上的差异来惩罚不光滑的决策图。

    Args:
        decision_map: (B, N, 1, H, W) 或 (B, N, H, W) 决策概率/权重
    Returns:
        scalar 损失值
    """
    # 去除最后一维（如果存在），变成 (B, N, H, W)
    prob = decision_map.squeeze(2)  # (B, N, H, W)
    # 计算水平方向相邻像素的绝对差
    diff_x = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
    # 计算垂直方向相邻像素的绝对差
    diff_y = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
    # 返回平均差值，越小越平滑
    return (diff_x.mean() + diff_y.mean()) / 2
