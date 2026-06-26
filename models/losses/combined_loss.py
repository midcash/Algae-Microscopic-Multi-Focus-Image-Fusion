"""
R17 损失：SSIM + GradMatch + GradContrast + TV + 伪标签 Focal Loss + Edge Magnitude Consistency

R17a: 在 R16 基础上，新增 edge_mag consistency loss，约束融合图梯度幅值接近选中源的梯度幅值。
目标：提升标准 QABF（对边缘传递一致性敏感）。

R22 (auto-experiment): 新增 Edge-Aware TV Loss — 在边缘区域放宽TV约束，仅在平坦区域施加平滑，
解决硬决策低置信区碎片化问题，同时保持边缘锐利。

Loss = SSIM(1.0) + GradMatch(0.5) + GradContrast(1.0) + TV(0.05) + Focal(可选权重) + EdgeMag(可选权重)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.decision_net import _gradient_magnitude, total_variation_loss


def gaussian_kernel(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)
    return kernel.unsqueeze(0).unsqueeze(0)


def _ssim(img1, img2, kernel_size=11, sigma=1.5):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    kernel = gaussian_kernel(kernel_size, sigma).to(img1.device)
    kernel = kernel.repeat(img1.shape[1], 1, 1, 1)
    pad = kernel_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=img2.shape[1])
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return (1 - ssim_map).mean()


def _gradient(img):
    """多通道 Sobel 梯度幅值"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(img.device)
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(img.shape[1], 1, 1, 1)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(img.shape[1], 1, 1, 1)
    gx = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    gy = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)


def compute_pseudo_labels(sources, gap_thresh=0.2, temperature=0.3, window_size=0):
    """
    多尺度区域一致性 Laplacian 伪标签

    1. 逐像素计算多尺度 Laplacian 响应 → 归一化 → softmax
    2. 对 argmax 做 mode filter（多数投票），实现区域一致性
    3. mode filter 后的硬标签 + 原始软标签混合，保留置信度信息
    4. gap 门控：低置信区域回退到均匀分布

    论文描述: "多尺度区域一致性 Laplacian 伪标签" — 在逐像素清晰度
    基础上引入局部窗口多数投票，使相邻像素对最优源图的选择达成共识。

    Args:
        sources: List[Tensor] (B,3,H,W)
        gap_thresh: top-2 gap阈值
        temperature: softmax温度
        window_size: mode filter窗口大小 (0=不启用, 建议5)
    Returns: (B, 5, H, W) 软伪标签
    """
    B = sources[0].shape[0]
    H, W = sources[0].shape[2:]
    device = sources[0].device
    N = len(sources)
    lap_k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)

    # Step 1: 多尺度 Laplacian
    scores = []
    for scale in [1.0, 0.5]:
        if scale < 1.0:
            scaled = [F.interpolate(s, (int(H*scale), int(W*scale)), mode='bilinear', align_corners=False) for s in sources]
        else:
            scaled = sources
        grads = []
        for s in scaled:
            gray = s.mean(dim=1, keepdim=True)
            lap = F.conv2d(gray, lap_k, padding=1)
            grads.append(lap.abs())
        ss = torch.cat(grads, dim=1)
        if scale < 1.0:
            ss = F.interpolate(ss, (H, W), mode='bilinear', align_corners=False)
        scores.append(ss)

    g = sum(scores) / len(scores)  # (B, N, H, W)

    # Step 2: 归一化 + softmax
    g_min = g.min(dim=1, keepdim=True)[0]
    g_max = g.max(dim=1, keepdim=True)[0]
    g_norm = (g - g_min) / (g_max - g_min + 1e-8)
    pseudo_sharp = F.softmax(g_norm / temperature, dim=1)  # (B, N, H, W)

    # Step 3: 区域一致性 — mode filter on argmax
    if window_size > 1:
        hard_idx = pseudo_sharp.argmax(dim=1)  # (B, H, W)
        # 转为 one-hot 做窗口投票
        onehot = F.one_hot(hard_idx, num_classes=N).permute(0, 3, 1, 2).float()  # (B, N, H, W)
        pad = window_size // 2
        vote = F.avg_pool2d(onehot, kernel_size=window_size, stride=1, padding=pad)
        if vote.shape[2] != H or vote.shape[3] != W:
            vote = F.interpolate(vote, size=(H, W), mode='bilinear', align_corners=False)
        # 多数投票结果作为 soft label（平滑过渡，不硬切换）
        regional_label = F.softmax(vote / 0.1, dim=1)  # 低温 → 接近 one-hot

        # 混合：原始软标签 × 0.3 + 区域一致性标签 × 0.7
        pseudo_sharp = 0.3 * pseudo_sharp + 0.7 * regional_label

    # Step 4: gap 门控
    top2 = torch.topk(pseudo_sharp, k=2, dim=1)
    gap = top2[0][:, 0] - top2[0][:, 1]
    uncertain_mask = (gap < gap_thresh).float().unsqueeze(1)
    pseudo_uniform = torch.full_like(pseudo_sharp, 1.0 / N)
    pseudo = pseudo_sharp * (1 - uncertain_mask) + pseudo_uniform * uncertain_mask

    return pseudo


def compute_laplacian_oracle_labels(sources, window_size=8):
    """
    Laplacian Oracle 硬标签：区域窗口内多数投票决定最优源图

    1. 对每张源图计算 Laplacian 响应
    2. 在 window_size × window_size 窗口内求和
    3. 窗口内所有像素共享同一个最优源图标签（argmax）
    4. 上采样回原始分辨率

    论文描述: "以局部区域 Laplacian 能量作为清晰度准则，生成区域一致性训练标签"

    Returns:
        oracle_labels: (B, H, W) LongTensor，每个像素的最优源图索引
        confidence: (B, H, W) FloatTensor，top1/(top1+top2) 作为置信度
    """
    B = sources[0].shape[0]
    H, W = sources[0].shape[2:]
    device = sources[0].device
    N = len(sources)
    lap_k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)

    # 每源 Laplacian 绝对值
    laps = []
    for src in sources:
        gray = src.mean(dim=1, keepdim=True)
        lap = F.conv2d(gray, lap_k, padding=1).abs()
        laps.append(lap)
    lap_stack = torch.cat(laps, dim=1)  # (B, N, H, W)

    # 区域窗口投票
    if window_size > 1:
        # 窗口内求和 → 区域清晰度
        pad = window_size // 2
        lap_pool = F.avg_pool2d(lap_stack, kernel_size=window_size, stride=1, padding=pad)
        if lap_pool.shape[2] != H or lap_pool.shape[3] != W:
            lap_pool = F.interpolate(lap_pool, size=(H, W), mode='bilinear', align_corners=False)
    else:
        lap_pool = lap_stack

    # Argmax → 每个像素选最优源图
    oracle_labels = lap_pool.argmax(dim=1)  # (B, H, W)

    # 置信度：top1 / (top1 + top2)
    top2 = torch.topk(lap_pool, k=2, dim=1)
    confidence = top2.values[:, 0] / (top2.values[:, 0] + top2.values[:, 1] + 1e-8)  # (B, H, W)

    return oracle_labels, confidence


class OracleGuidedLoss(nn.Module):
    """
    Oracle 引导损失 — 用 Laplacian Oracle 硬标签训练 DecisionNet

    高置信区域: CE Loss 强制选 oracle 指定的源图
    低置信区域: 允许弱 CE，减少强制

    Args:
        oracle_weight: oracle loss 权重
        focal_gamma: 低置信区域的 focal 衰减
        window_size: oracle 区域窗口大小
    """
    def __init__(self, oracle_weight=1.0, focal_gamma=1.0, window_size=8):
        super().__init__()
        self.oracle_weight = oracle_weight
        self.focal_gamma = focal_gamma
        self.window_size = window_size

    def forward(self, logits, sources):
        """
        Args:
            logits: (B, N, H, W) DecisionNet logits
            sources: List[Tensor] 源图
        Returns:
            loss: scalar
        """
        labels, confidence = compute_laplacian_oracle_labels(sources, self.window_size)

        # CE loss per pixel
        ce = F.cross_entropy(logits, labels, reduction='none')  # (B, H, W)

        # 高置信区域：全量 CE；低置信区域：衰减 CE
        # confidence 越高 → loss 权重越大
        weight = confidence.pow(self.focal_gamma)

        loss = (ce * weight).mean()
        return loss


class FocalLoss(nn.Module):
    """
    多分类 Focal Loss — 用于像素级伪标签监督
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 5, H, W) — DecisionNet 原始 logits
            targets: (B, 5, H, W) — soft 伪标签（概率分布）
        Returns:
            loss: scalar
        """
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs.clamp(1e-10, 1.0))
        focal_weight = (1 - probs).pow(self.gamma)
        loss = -(targets * focal_weight * log_probs).sum(dim=1)
        return loss.mean()


def edge_magnitude_consistency(fused, sources, decision_map):
    """
    R17a: Edge Magnitude Consistency Loss

    约束融合图在每像素处的 Sobel 梯度幅值，接近被选中源图在该像素的梯度幅值。
    用 decision_map 做加权聚合，而不是 argmax，保持可微。

    Args:
        fused: (B, 3, H, W) 融合图
        sources: List[Tensor] (B, 3, H, W) * N 源图
        decision_map: (B, N, 1, H, W) — Gumbel softmax 决策图
    Returns:
        loss: scalar
    """
    N = len(sources)

    # 融合图梯度幅值 (B, 3, H, W) -> (B, 1, H, W) [取灰度]
    fused_gray = fused.mean(dim=1, keepdim=True)
    grad_f = _gradient(fused_gray)  # (B, 1, H, W)

    # 各源图梯度幅值
    src_grads = []
    for src in sources:
        gray = src.mean(dim=1, keepdim=True)
        src_grads.append(_gradient(gray))  # (B, 1, H, W)

    # 决策图加权聚合选中的源图梯度
    # decision_map: (B, N, 1, H, W) -> squeeze dim2 -> (B, N, H, W) -> unsqueeze dim1 -> (B, 1, N, H, W)
    # stacked_grads: (B, N, 1, H, W) -> squeeze dim2 -> (B, N, H, W) -> unsqueeze dim1 -> (B, 1, N, H, W)
    # 但 L1 loss 需要 (B, 1, H, W)，所以用 einsum 或加权求和

    decision_s = decision_map.squeeze(2)  # (B, N, H, W)
    grad_stack = torch.stack(src_grads, dim=1)  # (B, N, 1, H, W)
    selected_grad = (decision_s.unsqueeze(2) * grad_stack).sum(dim=1)  # (B, 1, H, W)

    return F.l1_loss(grad_f, selected_grad)


def _sobel_orientation(img):
    """
    计算 Sobel 梯度方向角（与 QABF 的 getArray 一致）。

    Args:
        img: (B, 1, H, W) 灰度图
    Returns:
        g: (B, 1, H, W) 梯度幅值
        a: (B, 1, H, W) 方向角 [-π/2, π/2]
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(img.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).to(img.device)
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)

    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-10)

    # 方向角：arctan(gy/gx)，注意 gy/gx 的符号
    a = torch.atan2(gy + 1e-10, gx + 1e-10)  # (-π, π]
    return g, a


def _orientation_consistency(fused, sources, decision_map, grad_threshold=0.7):
    """
    R17b: Orientation Consistency Loss — 仅在强边缘区域计算

    QABF 的 QaAF 项：AAF = 1 - |aF - aA| / (π/2)。
    loss 只在选中源图边缘幅值排在 top (1 - grad_threshold) 的区域计算，
    避免平坦区域（方向随机）和弱边缘区域淹没梯度。

    Args:
        fused: (B, 3, H, W)
        sources: List[Tensor] (B, 3, H, W)
        decision_map: (B, N, 1, H, W)
        grad_threshold: float — 只在前 (1-grad_threshold) 分位的强边缘计算
                        如 0.7 → 前 30% 强边缘
    Returns:
        loss: scalar
    """
    N = len(sources)

    # 提取灰度方向
    def _gray_orient(img):
        gray = img.mean(dim=1, keepdim=True)
        g, a = _sobel_orientation(gray)
        return g, a

    g_fused, a_fused = _gray_orient(fused)  # (B, 1, H, W)

    src_oris, src_mags = [], []
    for src in sources:
        g_src, a_src = _gray_orient(src)
        src_oris.append(a_src)
        src_mags.append(g_src)

    # 选中源的方向角和边缘幅值
    stacked_oris = torch.stack(src_oris, dim=1)  # (B, N, 1, H, W)
    stacked_mags = torch.stack(src_mags, dim=1)  # (B, N, 1, H, W)
    decision_flat = decision_map.squeeze(2)  # (B, N, H, W)

    selected_ori = (decision_flat.unsqueeze(2) * stacked_oris).sum(dim=1)  # (B, 1, H, W)
    selected_mag = (decision_flat.unsqueeze(2) * stacked_mags).sum(dim=1)  # (B, 1, H, W)

    # 方向角差，wrap 到 [-π, π]
    diff = a_fused - selected_ori
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))

    # 梯度门控：只在前 (1 - grad_threshold) 分位的强边缘区域计算
    # 对每个 batch 各自找分位值
    B = fused.shape[0]
    total_loss = 0.0
    total_pixels = 0

    for b in range(B):
        mag_b = selected_mag[b, 0]  # (H, W)
        diff_b = diff[b, 0]  # (H, W)

        if mag_b.numel() < 10 or mag_b.max() < 1e-6:
            continue

        # 找到前 (1 - grad_threshold) 分位的阈值
        quantile_val = torch.quantile(mag_b, grad_threshold)  # 如 0.7 → 第 70% 分位的值
        mask = mag_b > quantile_val  # 取前 30%
        n_strong = mask.sum()

        if n_strong > 0:
            loss_b = (torch.abs(diff_b) * mask).sum() / (torch.pi / 2)
            total_loss += loss_b / n_strong
            total_pixels += 1

    return total_loss / max(total_pixels, 1)


def _decoder_consistency_loss(decoder_features, sources, proj_weight):
    """
    Step 2: decoder 特征与 Laplacian 清晰度的一致性 loss。

    核心思想：
    - decoder_features (B, C=8, H, W) 编码每个位置的清晰度特征
    - 用固定的随机投影（每 epoch 固定）或可学习的投影将 C 维映射到 N 通道
    - 与 Laplacian 清晰度分数做 KL 散度

    Args:
        decoder_features: (B, C, H, W) — decoder 输出
        sources: List[Tensor] (N, B, 3, H, W)
        proj_weight: (C, N) — 固定的投影矩阵
    Returns:
        loss: scalar
    """
    N = len(sources)
    B, C, H, W = decoder_features.shape
    device = decoder_features.device

    # 1. Laplacian 清晰度分数
    lap_k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)
    lap_scores = []
    for s in sources:
        gray = s.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        lap = F.conv2d(gray, lap_k, padding=1)
        lap_scores.append(lap.abs())
    lap_stack = torch.stack(lap_scores, dim=1).squeeze(2)  # (B, N, H, W)

    # 每个像素在 N 张图上的分数做 softmax
    lap_probs = F.softmax(lap_stack, dim=1).detach()

    # 2. decoder_features 投影到 N 通道 (1x1 conv 等价)
    # decoder_feat: (B, C, H, W) → permute → (B, H, W, C) → matmul proj → (B, H, W, N)
    dec_flat = decoder_features.permute(0, 2, 3, 1)  # (B, H, W, C)
    dec_proj = dec_flat @ proj_weight  # (B, H, W, N)
    dec_proj = dec_proj.permute(0, 3, 1, 2)  # (B, N, H, W)
    dec_probs = F.softmax(dec_proj, dim=1)

    # 3. KL 散度
    kl = (lap_probs * (torch.log(lap_probs + 1e-10) - torch.log(dec_probs + 1e-10))).sum(dim=1)
    return kl.mean()


def _low_confidence_consistency_loss(logits, gap_thresh=0.15, kernel_size=3):
    """
    R20-v3-lite: 仅对低置信区域施加局部一致性约束。

    思想：对 softmax(logits) 得到的决策概率图做局部平均，
    仅在 top1-top2 gap 较小的区域，让当前概率分布贴近局部均值，
    从而抑制低置信区域的随机碎片化，而不压制高置信边缘。
    """
    probs = F.softmax(logits, dim=1)  # (B, N, H, W)
    top2_vals, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    top1 = top2_vals[:, 0:1]
    top2 = top2_vals[:, 1:2] if top2_vals.shape[1] > 1 else torch.zeros_like(top1)
    gap = top1 - top2
    low_conf_mask = (gap < gap_thresh).float()  # (B,1,H,W)

    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1

    local_mean = F.avg_pool2d(
        probs.reshape(-1, 1, probs.shape[2], probs.shape[3]),
        kernel_size=k,
        stride=1,
        padding=k // 2,
    ).reshape_as(probs)

    diff = (probs - local_mean).abs().sum(dim=1, keepdim=True)
    denom = low_conf_mask.sum() + 1e-8
    return (diff * low_conf_mask).sum() / denom


def _edge_aware_tv_loss(decision_map, fused_img, edge_threshold=0.1, edge_decay=3.0):
    """
    R22 (auto-experiment): Edge-Aware Total Variation Loss

    核心思想：标准TV loss无差别平滑所有区域，会损失边缘锐利度。
    通过融合图像的梯度幅值作为门控信号：
    - 高梯度区（边缘）：TV惩罚衰减，保留锐利边缘
    - 低梯度区（平坦/碎片）：全量TV惩罚，消除噪声碎片

    Args:
        decision_map: (B, N, 1, H, W) 决策权重图
        fused_img: (B, 3, H, W) 融合图像，用于计算边缘门控
        edge_threshold: 梯度幅值阈值，大于此值的区域视为边缘
        edge_decay: 边缘区域TV权重的衰减指数（越大衰减越快）

    Returns:
        scalar: edge-aware TV loss
    """
    # 1. 计算融合图像的梯度幅值（边缘强度）
    fused_gray = fused_img.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    gx = F.conv2d(fused_gray, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                            dtype=torch.float32, device=fused_img.device), padding=1)
    gy = F.conv2d(fused_gray, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                            dtype=torch.float32, device=fused_img.device), padding=1)
    edge_map = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)  # (B, 1, H, W)

    # 2. 归一化到[0,1]
    edge_map = edge_map / (edge_map.max() + 1e-8)

    # 3. 边缘门控权重：边缘 → 低权重(不惩罚TV), 平坦 → 高权重(惩罚碎片)
    # weight = exp(-edge_decay * edge_map / edge_threshold)
    edge_weight = torch.exp(-edge_decay * edge_map / (edge_threshold + 1e-8))  # (B, 1, H, W)

    # 确保空间尺寸匹配
    if edge_weight.shape[2:] != decision_map.shape[3:]:
        edge_weight = F.interpolate(edge_weight, size=decision_map.shape[3:],
                                     mode='bilinear', align_corners=False)

    # 4. 计算决策图的空间梯度（h, w方向）
    prob = decision_map.squeeze(2)  # (B, N, H, W)
    diff_h = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])  # (B, N, H-1, W)
    diff_w = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])  # (B, N, H, W-1)

    # 5. 应用边缘门控到TV
    # 对h方向: edge_weight需要匹配diff_h的空间尺寸
    edge_h = F.interpolate(edge_weight, size=diff_h.shape[2:],
                           mode='bilinear', align_corners=False) if edge_weight.shape[2:] != diff_h.shape[2:] else edge_weight
    edge_w = F.interpolate(edge_weight, size=diff_w.shape[2:],
                           mode='bilinear', align_corners=False) if edge_weight.shape[2:] != diff_w.shape[2:] else edge_weight

    # 广播到N通道
    edge_h = edge_h.expand(-1, diff_h.shape[1], -1, -1)
    edge_w = edge_w.expand(-1, diff_w.shape[1], -1, -1)

    tv_h = (diff_h * edge_h).mean()
    tv_w = (diff_w * edge_w).mean()

    return (tv_h + tv_w) / 2


def _bilateral_smooth_loss(decision_map, fused_img, sigma_spatial=1.0, sigma_range=0.1):
    """
    R22 (auto-experiment): Bilateral Smoothness Loss

    用双边滤波的思想平滑决策图：在同一物体表面（颜色相近）的区域，决策应该一致。
    相比edge-gated TV，双边平滑考虑的是色彩相似性而非仅梯度。

    简化的可微实现：用fused_img的颜色距离对决策差异加权。

    Args:
        decision_map: (B, N, 1, H, W)
        fused_img: (B, 3, H, W)
        sigma_spatial: 空间高斯sigma
        sigma_range: 颜色高斯sigma

    Returns:
        scalar loss
    """
    prob = decision_map.squeeze(2)  # (B, N, H, W)

    # 水平/垂直差异
    diff_h = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
    diff_w = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])

    # 颜色距离（用平均池化模拟双边滤波的颜色项）
    fused_norm = fused_img / (fused_img.norm(dim=1, keepdim=True) + 1e-8)
    color_diff_h = (fused_norm[:, :, 1:, :] - fused_norm[:, :, :-1, :]).norm(dim=1, keepdim=True)
    color_diff_w = (fused_norm[:, :, :, 1:] - fused_norm[:, :, :, :-1]).norm(dim=1, keepdim=True)

    # 颜色相似度 → 权重（颜色越相似，越应该平滑）
    # 即：颜色相似区域的决策差异应该小
    w_h = torch.exp(-color_diff_h / (2 * sigma_range ** 2)).squeeze(1)  # (B, H-1, W)
    w_w = torch.exp(-color_diff_w / (2 * sigma_range ** 2)).squeeze(1)  # (B, H, W-1)

    # 广播
    w_h = w_h.unsqueeze(1).expand(-1, diff_h.shape[1], -1, -1)
    w_w = w_w.unsqueeze(1).expand(-1, diff_w.shape[1], -1, -1)

    loss_h = (diff_h * w_h).mean()
    loss_w = (diff_w * w_w).mean()

    return (loss_h + loss_w) / 2


class CombinedLoss(nn.Module):
    """
    Step 2: Combined Loss — 统一 Loss 接口，加入 decoder_features 监督

    核心架构：
    - SSIM(权重) — 融合图与各源图的结构相似度
    - GradMatch(权重) — 融合图梯度与各源图梯度的 L1 匹配
    - GradContrast(权重) — 选中的源图梯度应 > 未选中的
    - TV(权重) — 决策图空间平滑
    - Focal(权重) — 伪标签直接监督 DecisionNet logits
    - EdgeMag(权重) — 融合图边缘强度与选中源图的一致性
    - Orientation(权重) — 融合图边缘方向与选中源图的一致性
    - DecoderConsistency(权重) — Step 2: decoder_features 与 Laplacian 清晰度的 KL 一致性

    模型 forward 始终返回 (fused, decision_map, logits) 三件套。
    如果需要 decoder_features 参与 loss，在模型 forward 中返回 4 件套 (fused, decision_map, logits, decoder_feat)，
    或在外部通过额外的参数传入。

    Args:
        margin: 梯度对比损失的 margin
        pseudo_label_weight: 伪标签 Focal Loss 权重（设为 >0 时启用）
        focal_gamma: Focal Loss 的 gamma 参数
        gap_thresh: 伪标签 top-2 gap 阈值
        pseudo_temperature: 伪标签 softmax 温度
        edge_mag_weight: Edge Magnitude Consistency 权重
        orientation_weight: Orientation Consistency 权重
        decoder_consistency_weight: Step 2 新增 — decoder 特征与 Laplacian 清晰度的一致性权重
    """
    def __init__(self, ssim_weight=1.0, grad_match_weight=0.5,
                 grad_contrast_weight=1.0, tv_weight=0.05, margin=0.05,
                 pseudo_label_weight=0.0, focal_gamma=2.0,
                 gap_thresh=0.2, pseudo_temperature=0.3,
                 pseudo_window_size=0,
                 oracle_weight=0.0, oracle_window_size=8,
                 edge_mag_weight=0.0, orientation_weight=0.0,
                 decoder_consistency_weight=0.0,
                 low_conf_consistency_weight=0.0,
                 low_conf_gap_thresh=0.15,
                 low_conf_kernel_size=3,
                 edge_aware_tv_weight=0.0,
                 edge_aware_tv_threshold=0.1,
                 edge_aware_tv_decay=3.0,
                 bilateral_smooth_weight=0.0,
                 bilateral_sigma_range=0.1):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.grad_match_weight = grad_match_weight
        self.grad_contrast_weight = grad_contrast_weight
        self.tv_weight = tv_weight
        self.margin = margin
        self.pseudo_label_weight = pseudo_label_weight
        self.focal_gamma = focal_gamma
        self.gap_thresh = gap_thresh
        self.pseudo_temperature = pseudo_temperature
        self.pseudo_window_size = pseudo_window_size
        self.oracle_weight = oracle_weight
        self.oracle_window_size = oracle_window_size
        if oracle_weight > 0:
            self.oracle_loss = OracleGuidedLoss(
                oracle_weight=oracle_weight, focal_gamma=1.0, window_size=oracle_window_size)
        self.edge_mag_weight = edge_mag_weight
        self.orientation_weight = orientation_weight
        self.decoder_consistency_weight = decoder_consistency_weight
        self.low_conf_consistency_weight = low_conf_consistency_weight
        self.low_conf_gap_thresh = low_conf_gap_thresh
        self.low_conf_kernel_size = low_conf_kernel_size
        self.ori_grad_threshold = 0.7
        # R22: Edge-aware TV
        self.edge_aware_tv_weight = edge_aware_tv_weight
        self.edge_aware_tv_threshold = edge_aware_tv_threshold
        self.edge_aware_tv_decay = edge_aware_tv_decay
        # R22: Bilateral smooth
        self.bilateral_smooth_weight = bilateral_smooth_weight
        self.bilateral_sigma_range = bilateral_sigma_range
        if pseudo_label_weight > 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma)
        # Step 2: decoder 投影矩阵 — 固定随机初始化的可学习参数
        if decoder_consistency_weight > 0:
            decoder_feat_channels = 8  # MSegNetV2 decoder 尾部通道数
            self.decoder_proj_weight = nn.Parameter(
                torch.randn(decoder_feat_channels, 5) * 0.01,
            )
        else:
            self.decoder_proj_weight = None

    def forward(self, fused, sources, decision_map, logits, decoder_features=None):
        """
        Args:
            fused: (B, 3, H, W)
            sources: List[Tensor] (N, B, 3, H, W)
            decision_map: (B, N, 1, H, W) — one-hot 或软权重决策图
            logits: (B, N, H, W) — 原始 logits
            decoder_features: (B, C=8, H, W) — Step 2 新增，可选
        """
        N = len(sources)
        B, _, H, W = fused.shape
        device = fused.device

        # 1. SSIM
        ssim = sum(_ssim(fused, src) for src in sources) / N

        # 2. GradMatch
        fused_grad = _gradient(fused)
        src_grads = [_gradient(src) for src in sources]
        grad_match = sum(F.l1_loss(fused_grad, g) for g in src_grads) / N

        # 3. GradContrast
        src_grad_mags = [_gradient_magnitude(src) for src in sources]
        stacked_grads = torch.stack(src_grad_mags, dim=1)  # (B, N, 1, H, W)
        selected_grad = (decision_map * stacked_grads).sum(dim=1, keepdim=True)
        selected_grad = selected_grad.squeeze(1)  # (B, 1, H, W)

        contrast_loss = 0.0
        count = 0
        for i in range(N):
            not_selected = (decision_map[:, i:i+1] < 0.5).float()
            if not_selected.sum() > 0:
                not_sel_grad = src_grad_mags[i]
                diff = not_sel_grad - selected_grad + self.margin
                hinge = torch.relu(diff)
                not_sel_mask = not_selected.squeeze(1)
                contrast_loss += (hinge * not_sel_mask).sum() / (not_sel_mask.sum() + 1e-8)
                count += 1
        if count > 0:
            contrast_loss = contrast_loss / count
        else:
            contrast_loss = 0.0

        # 4. TV 平滑
        tv = total_variation_loss(decision_map)

        # 5. R16: 伪标签 Focal Loss（可选）
        pseudo_loss = 0.0
        if self.pseudo_label_weight > 0 and hasattr(self, 'focal_loss'):
            targets = compute_pseudo_labels(
                sources,
                gap_thresh=self.gap_thresh,
                temperature=self.pseudo_temperature,
                window_size=self.pseudo_window_size,
            )
            pseudo_loss = self.focal_loss(logits, targets)

        # 5b. Oracle Guided Loss — 直接用 Laplacian Oracle 硬标签训练
        oracle_loss_val = 0.0
        if self.oracle_weight > 0 and hasattr(self, 'oracle_loss'):
            oracle_loss_val = self.oracle_loss(logits, sources)

        # 6. R17a: Edge Magnitude Consistency（可选）
        edge_mag = 0.0
        if self.edge_mag_weight > 0:
            edge_mag = edge_magnitude_consistency(fused, sources, decision_map)

        # 7. R17b: Orientation Consistency（可选）— 直接对齐 QABF 的 QaAF 方向项
        ori_loss = 0.0
        if self.orientation_weight > 0:
            ori_loss = _orientation_consistency(fused, sources, decision_map,
                                                grad_threshold=self.ori_grad_threshold)

        # 8. Step 2: Decoder Consistency（可选）— decoder_features 与 Laplacian 清晰度一致
        dec_cons = 0.0
        if self.decoder_consistency_weight > 0 and decoder_features is not None and self.decoder_proj_weight is not None:
            dec_cons = _decoder_consistency_loss(decoder_features, sources, self.decoder_proj_weight)

        # 9. R20-v3-lite: 低置信区域局部一致性约束
        low_conf_cons = 0.0
        if self.low_conf_consistency_weight > 0:
            low_conf_cons = _low_confidence_consistency_loss(
                logits,
                gap_thresh=self.low_conf_gap_thresh,
                kernel_size=self.low_conf_kernel_size,
            )

        # 10. R22: Edge-Aware TV — 边缘区域放宽TV，平坦区域惩罚碎片
        edge_aware_tv = 0.0
        if self.edge_aware_tv_weight > 0:
            edge_aware_tv = _edge_aware_tv_loss(
                decision_map, fused,
                edge_threshold=self.edge_aware_tv_threshold,
                edge_decay=self.edge_aware_tv_decay,
            )

        # 11. R22: Bilateral Smoothness — 颜色相似的相邻像素决策应一致
        bilateral_smooth = 0.0
        if self.bilateral_smooth_weight > 0:
            bilateral_smooth = _bilateral_smooth_loss(
                decision_map, fused,
                sigma_spatial=1.0,
                sigma_range=self.bilateral_sigma_range,
            )

        total = (self.ssim_weight * ssim +
                 self.grad_match_weight * grad_match +
                 self.grad_contrast_weight * contrast_loss +
                 self.tv_weight * tv +
                 self.pseudo_label_weight * pseudo_loss +
                 self.edge_mag_weight * edge_mag +
                 self.orientation_weight * ori_loss +
                 self.decoder_consistency_weight * dec_cons +
                 self.low_conf_consistency_weight * low_conf_cons +
                 self.edge_aware_tv_weight * edge_aware_tv +
                 self.bilateral_smooth_weight * bilateral_smooth +
                 self.oracle_weight * oracle_loss_val)

        self._last_details = {
            'ssim': ssim.item(),
            'grad_match': grad_match.item(),
            'contrast': contrast_loss if isinstance(contrast_loss, float) else contrast_loss.item(),
            'tv': tv.item(),
            'pseudo_focal': pseudo_loss if isinstance(pseudo_loss, float) else pseudo_loss.item(),
            'edge_mag': edge_mag if isinstance(edge_mag, float) else edge_mag.item(),
            'orient': ori_loss if isinstance(ori_loss, float) else ori_loss.item(),
            'decoder_cons': dec_cons if isinstance(dec_cons, float) else dec_cons.item(),
            'low_conf_cons': low_conf_cons if isinstance(low_conf_cons, float) else low_conf_cons.item(),
            'edge_aware_tv': edge_aware_tv if isinstance(edge_aware_tv, float) else edge_aware_tv.item(),
            'bilateral_smooth': bilateral_smooth if isinstance(bilateral_smooth, float) else bilateral_smooth.item(),
            'oracle_loss': oracle_loss_val if isinstance(oracle_loss_val, float) else oracle_loss_val.item(),
        }
        return total
