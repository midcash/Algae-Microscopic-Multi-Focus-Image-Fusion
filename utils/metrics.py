"""
评估指标计算

支持多种图像融合质量评估指标。

实现说明：
- QABF: Xydeas & Petrovic (2000) 标准边缘信息保留度（基于用户提供的生产级实现）
- MI: 5源图全面平均（原实现只取前2张）
- SF / AG: 正确的二维梯度计算
- PSNR / SSIM: 标准实现（但GT不可信，仅供参考）
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import convolve2d


# ==================== QABF (Xydeas & Petrovic 2000) ====================

def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.
    """
    hwsize = (wsize - 1) // 2
    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext


def sobel_fn(x):
    """Sobel gradient decomposition (separate gv, gh)."""
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    return gv, gh


def get_Qabf(pA, pB, pF):
    """
    标准 QABF 边缘信息保留度 (Xydeas & Petrovic 2000)

    衡量融合图像 pF 从源图 pA, pB 中保留的边缘信息量。

    Args:
        pA, pB: 源图像 (H, W), uint8 或 float
        pF: 融合图像 (H, W), uint8 或 float

    Returns:
        QABF 值 [0, 1], 越高越好
    """
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel operators for gradient + orientation
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    def flip180(arr):
        return np.flip(arr)

    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        return convolve2d(data, k, mode='valid')

    def getArray(img):
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        g = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        a = np.zeros((n, m))
        zero_mask = SAx == 0
        a[~zero_mask] = np.arctan(SAy[~zero_mask] / SAx[~zero_mask])
        a[zero_mask] = np.pi / 2
        return g, a

    gA, aA = getArray(pA)
    gB, aB = getArray(pB)
    gF, aF = getArray(pF)

    def getQabf_single(aA, gA, aF, gF):
        mask = (gA > gF)
        GAF = np.where(mask, gF / (gA + 1e-10), np.where(gA == gF, gF, gF / (gA + 1e-10)))
        AAF = 1 - np.abs(aA - aF) / (np.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    QAF = getQabf_single(aA, gA, aF, gF)
    QBF = getQabf_single(aB, gB, aF, gF)

    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)
    return nume / (deno + 1e-10)


# ==================== 全参考指标 ====================

def psnr(img1, img2, max_value=1.0):
    """峰值信噪比 (PSNR) — 但 GT 不可信，仅供参考。"""
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')

    psnr_val = 20 * np.log10(max_value) - 10 * torch.log10(mse)
    return psnr_val.item()


def ssim(img1, img2, kernel_size=11, sigma=1.5):
    """结构相似性 (SSIM) — 但 GT 不可信，仅供参考。"""
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).unsqueeze(0)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0).to(img1.device)
    kernel = kernel.repeat(img1.shape[1], 1, 1, 1)

    mu1 = F.conv2d(img1, kernel, padding=kernel_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=kernel_size//2, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# ==================== 无参考指标 ====================

def spatial_frequency(img):
    """
    空间频率 (Spatial Frequency)

    正确的二维实现：先展平到单通道灰度，再计算行/列均方根梯度。
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0, 0]
    elif img.ndim == 3:
        img = img[0]

    row_grad = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    col_grad = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    sf = np.sqrt(row_grad ** 2 + col_grad ** 2)
    return float(sf)


def average_gradient(img):
    """
    平均梯度 (Average Gradient)

    修复：使用正确的二维 Sobel 卷积（scipy.signal.convolve2d），
    不再用 np.convolve 的一维展平错误做法。
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0, 0]
    elif img.ndim == 3:
        img = img[0]

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve2d(img, sobel_x, mode='same')
    grad_y = convolve2d(img, sobel_y, mode='same')

    ag = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))
    return float(ag)


def entropy(img):
    """信息熵 (Entropy) — 正确。"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0, 0]
    elif img.ndim == 3:
        img = img[0]

    img = np.clip((img * 255).astype(np.uint8), 0, 255)
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    hist = hist / hist.sum()
    ent = -np.sum(hist * np.log2(hist + 1e-10))
    return float(ent)


def mutual_information(fused, sources):
    """
    互信息 (Mutual Information)

    修复：接受 sources 列表，对所有源图两两平均，而非仅取前2张。

    Args:
        fused: 融合图像 (H, W)
        sources: 源图像列表，每张 (H, W)

    Returns:
        MI 值，越高越好
    """
    def _mi(img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        if img1.ndim == 4:
            img1 = img1[0, 0]
        if img2.ndim == 4:
            img2 = img2[0, 0]

        img1 = np.clip((img1 * 255).astype(np.uint8), 0, 255)
        img2 = np.clip((img2 * 255).astype(np.uint8), 0, 255)

        hist2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=64)
        hist2d = hist2d / hist2d.sum()

        px = hist2d.sum(axis=1, keepdims=True)
        py = hist2d.sum(axis=0, keepdims=True)

        mi = np.sum(hist2d * np.log2(hist2d / (px * py + 1e-10) + 1e-10))
        return float(mi)

    n = len(sources)
    total_mi = 0.0
    for i in range(n):
        total_mi += _mi(fused, sources[i])

    return total_mi / n  # 返回平均 MI


def qabf(fused, source1, source2):
    """
    QAB/F 边缘信息保留度（标准 Xydeas & Petrovic 2000 实现）

    使用 scipy.signal.convolve2d + 梯度幅值和方向联合评估。
    参考：用户提供的生产级 MATLAB->Python 移植代码。
    """
    if isinstance(fused, torch.Tensor):
        fused = fused.cpu().numpy()
    if isinstance(source1, torch.Tensor):
        source1 = source1.cpu().numpy()
    if isinstance(source2, torch.Tensor):
        source2 = source2.cpu().numpy()

    if fused.ndim == 4:
        fused = fused[0, 0]
    if source1.ndim == 4:
        source1 = source1[0, 0]
    if source2.ndim == 4:
        source2 = source2[0, 0]
    if fused.ndim == 3:
        fused = fused[0]
    if source1.ndim == 3:
        source1 = source1[0]
    if source2.ndim == 3:
        source2 = source2[0]

    return float(get_Qabf(source1, source2, fused))


def calculate_metrics(fused, sources=None, gt=None):
    """
    计算所有评估指标

    Args:
        fused: 融合图像 (tensor 或 numpy)
        sources: 源图像列表 (可选) — 用于 MI 和 QABF 全面平均
        gt: Ground Truth (可选) — 仅作参考，不可信

    Returns:
        dict: 包含所有指标
    """
    metrics = {}

    if gt is not None:
        metrics['psnr'] = psnr(fused, gt)
        metrics['ssim'] = ssim(fused, gt)

    metrics['spatial_frequency'] = spatial_frequency(fused)
    metrics['entropy'] = entropy(fused)

    if sources and len(sources) >= 2:
        metrics['average_gradient'] = average_gradient(fused)

        # MI: 所有源图的平均 MI
        sources_np = []
        for s in sources:
            if isinstance(s, torch.Tensor):
                s = s.cpu().numpy()
            if s.ndim == 4:
                s = s[0, 0]
            elif s.ndim == 3:
                s = s[0]
            sources_np.append(s)

        metrics['mutual_information'] = mutual_information(fused, sources_np)

        # QABF: 5源图全面两两平均
        qabf_vals = []
        n = len(sources_np)
        for i in range(n):
            for j in range(i + 1, n):
                qabf_vals.append(qabf(fused, sources_np[i], sources_np[j]))
        metrics['qabf'] = float(np.mean(qabf_vals)) if qabf_vals else 0.0

    return metrics


def evaluate_fusion(model, test_loader, device, num_samples=10):
    """评估模型在测试集上的表现。"""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            sources = [s.to(device) for s in batch['sources']]
            gt = batch.get('target', None)
            if gt is not None:
                gt = gt.to(device)

            fused = model(sources)

            sources_cpu = [s.cpu() for s in sources]
            fused_cpu = fused.cpu()
            gt_cpu = gt.cpu() if gt is not None else None

            metrics = calculate_metrics(fused_cpu, sources_cpu, gt_cpu)
            all_metrics.append(metrics)

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    return avg_metrics
