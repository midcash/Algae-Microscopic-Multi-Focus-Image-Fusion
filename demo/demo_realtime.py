"""
答辩实时演示 — 多聚焦图像融合现场运行

用法:
  python demo_realtime.py --group group_003                    # quality模式(全分辨率)
  python demo_realtime.py --group group_003 --mode fast        # fast模式(512x512)
  python demo_realtime.py --imgs a.png b.png c.png d.png e.png # 自定义图片

优化要点:
  - 批量分块推理: 多tile并行前向, 减少kernel launch开销
  - 向量化投票: np.add.at 替代逐超像素循环 O(n_sp*HW) -> O(HW)
  - 直接索引硬融合: mask索引替代broadcast乘法
  - 降采样羽化: 1/2分辨率计算alpha再upsample, 视觉无差异
"""

import argparse, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch
from skimage.segmentation import felzenszwalb, slic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.m_segnet_v5 import create_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_CKPT = str(ROOT / 'runs' / 'train' / 'auto_05_v5_raw_feature' / 'checkpoints' / 'best.pt')


# ============================================================
# 数据加载
# ============================================================

def load_sources_from_group(data_dir, group):
    srcs = []
    for i in range(1, 6):
        p = Path(data_dir) / group / f'img_{i}.png'
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(str(p))
        srcs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return srcs


def load_sources_from_files(file_paths):
    srcs = []
    for fp in file_paths:
        img = cv2.imread(fp)
        if img is None:
            raise FileNotFoundError(fp)
        srcs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return srcs


# ============================================================
# 模型推理
# ============================================================

def load_model(ckpt_path):
    model = create_model(num_source_images=5, use_fusion_head='gumbel', top_k=1)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(DEVICE).eval()
    return model


def _extract_tile_batch(srcs, tile_specs, tile_size):
    """
    从源图中提取一批 tile, 返回 List[Tensor] 每源图一个 (B, 3, tile, tile).
    tile_specs: [(y0, y1, x0, x1), ...]
    """
    B = len(tile_specs)
    batch_per_src = []
    for src in srcs:
        tiles = []
        for (y0, y1, x0, x1) in tile_specs:
            t = cv2.resize(src[y0:y1, x0:x1], (tile_size, tile_size))
            tiles.append(torch.from_numpy(t.astype(np.float32) / 255.0).permute(2, 0, 1))
        batch_per_src.append(torch.stack(tiles, dim=0).to(DEVICE))
    return batch_per_src


def tiled_inference_batched(model, srcs, tile_size=512, tile_batch=4):
    """全分辨率分块推理（批量并行）→ 全局决策图"""
    h, w = srcs[0].shape[:2]
    ty = (h + tile_size - 1) // tile_size
    tx = (w + tile_size - 1) // tile_size
    global_dec = np.zeros((h, w), dtype=np.int32)
    total_ms = 0.0
    n_total = ty * tx

    # 收集所有 tile 位置
    all_specs = []
    for yi in range(ty):
        for xi in range(tx):
            y0 = yi * tile_size; y1 = min(y0 + tile_size, h)
            x0 = xi * tile_size; x1 = min(x0 + tile_size, w)
            all_specs.append((y0, y1, x0, x1))

    processed = 0
    while processed < len(all_specs):
        batch_specs = all_specs[processed:processed + tile_batch]
        processed += len(batch_specs)

        # 构造 batch
        tiles = _extract_tile_batch(srcs, batch_specs, tile_size)

        t0 = time.perf_counter()
        with torch.no_grad():
            dm = model(tiles)[1]           # (B, N, 1, H, W)
            dec = dm.squeeze(2).argmax(dim=1).cpu().numpy()  # (B, tile, tile)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000

        # 将 batch 中每个 tile 的决策图拼回全局
        for i, (y0, y1, x0, x1) in enumerate(batch_specs):
            dec_i = dec[i]  # (tile, tile)
            dec_full = cv2.resize(dec_i.astype(np.float32), (x1 - x0, y1 - y0),
                                  interpolation=cv2.INTER_NEAREST)
            global_dec[y0:y1, x0:x1] = dec_full.astype(np.int32)

    return global_dec, total_ms, n_total


# ============================================================
# 后处理 — 向量化版本
# ============================================================

def segment_and_vote(dec_map, guide_img, felz_scale=200, felz_sigma=0.8, felz_min_size=200):
    """
    Felzenszwalb 分割 + 向量化多数投票.
    返回: dec_seg, seg, n_sp, seg_votes
    """
    h, w = dec_map.shape
    n_src = dec_map.max() + 1

    guide_small = cv2.resize(guide_img, (w // 4, h // 4))
    seg_small = felzenszwalb(guide_small, scale=felz_scale,
                             sigma=felz_sigma, min_size=felz_min_size)
    seg = cv2.resize(seg_small.astype(np.float32), (w, h),
                     interpolation=cv2.INTER_NEAREST).astype(np.int32)
    n_sp = seg.max() + 1

    flat_dec = dec_map.ravel()
    flat_seg = seg.ravel()
    seg_votes = np.zeros((n_sp, n_src), dtype=np.int32)
    np.add.at(seg_votes, (flat_seg, flat_dec), 1)

    winner = seg_votes.argmax(axis=1)
    dec_seg = winner[seg]

    return dec_seg, seg, n_sp, seg_votes


def merge_tiny_sp(dec_seg, seg, seg_votes, min_sp_size=500):
    """
    清理碎片SP: 小于 min_sp_size 的超像素, 将其决策替换为最近大SP的决策.
    从SP层面消除碎片, 而非从像素层面模糊.
    """
    h, w = dec_seg.shape
    n_sp = seg.max() + 1
    dec_new = dec_seg.copy()

    # 计算每个SP的大小
    sp_sizes = np.bincount(seg.ravel(), minlength=n_sp)

    # 找到每个SP的邻居SP（通过边界接触）
    # 构建邻接表: 扫描SP边界, 记录相邻SP对
    adjacency = {}
    for sp_id in range(n_sp):
        if sp_sizes[sp_id] >= min_sp_size:
            continue  # 只处理小SP
        # 膨胀小SP来找邻居
        mask = (seg == sp_id)
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3,3), np.uint8))
        neighbors = np.unique(seg[(dilated > 0) & (seg != sp_id)])
        adjacency[sp_id] = [(n, sp_sizes[n]) for n in neighbors]

    # 合并小SP: 采用最大邻居的源图决策
    n_merged = 0
    for sp_id, neighbors in sorted(adjacency.items(), key=lambda x: sp_sizes[x[0]]):
        if sp_sizes[sp_id] == 0:  # already absorbed
            continue
        if not neighbors:
            continue
        # 找最大的邻居
        best_neighbor = max(neighbors, key=lambda x: x[1])[0]
        winner_src = np.bincount(dec_new[seg == best_neighbor], minlength=5).argmax()
        dec_new[seg == sp_id] = winner_src
        n_merged += 1

    return dec_new, n_merged


def smooth_decision_boundary(dec_seg, guide_img, kernel_size=7):
    """
    引导滤波平滑决策边界: 以 guide_img 的灰度图为引导,
    在图像边缘处保持决策边界, 在平坦区平滑锯齿.
    引导滤波 O(N) 复杂度, 比高斯模糊更精准地保边.
    """
    h, w = dec_seg.shape
    n_src = dec_seg.max() + 1
    guide = cv2.cvtColor(guide_img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    r = kernel_size  # filter radius

    soft = np.zeros((h, w, n_src), dtype=np.float64)
    for c in range(n_src):
        mask = (dec_seg == c).astype(np.float64)
        # Guided filter: q = a*I + b, where I=guide, p=binary mask
        mean_I = cv2.boxFilter(guide, -1, (r, r), normalize=True)
        mean_p = cv2.boxFilter(mask, -1, (r, r), normalize=True)
        corr_I = cv2.boxFilter(guide * guide, -1, (r, r), normalize=True)
        corr_Ip = cv2.boxFilter(guide * mask, -1, (r, r), normalize=True)
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + 1e-4)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, (r, r), normalize=True)
        mean_b = cv2.boxFilter(b, -1, (r, r), normalize=True)
        soft[:, :, c] = (mean_a * guide + mean_b).clip(0, 1)

    # Argmax
    result = soft.argmax(axis=2).astype(np.int32)
    # Low confidence -> keep original
    max_val = soft.max(axis=2)
    result[max_val < 0.3] = dec_seg[max_val < 0.3]
    return result


def fuse_guided(dec_seg, srcs, guide_img, radius=15, bilateral_d=5,
                bilateral_sigma_color=30, bilateral_sigma_space=30):
    """
    引导滤波直接融合: 将SP投票后的硬决策转为引导滤波软权重,
    替代 硬融合+软融合+羽化 三步, 天然保边消锯齿. 最后+双边抗锯齿.
    """
    h, w = int(dec_seg.shape[0]), int(dec_seg.shape[1])
    n_src = len(srcs)
    srcs_f32 = [s.astype(np.float32) for s in srcs]

    # Step 1: Guided filter each source's binary mask -> soft weights
    guide = cv2.cvtColor(guide_img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    r = radius
    weights = np.zeros((h, w, n_src), dtype=np.float32)
    for c in range(n_src):
        mask = (dec_seg == c).astype(np.float64)
        mean_I = cv2.boxFilter(guide, -1, (r, r), normalize=True)
        mean_p = cv2.boxFilter(mask, -1, (r, r), normalize=True)
        corr_I = cv2.boxFilter(guide * guide, -1, (r, r), normalize=True)
        corr_Ip = cv2.boxFilter(guide * mask, -1, (r, r), normalize=True)
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + 1e-4)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, (r, r), normalize=True)
        mean_b = cv2.boxFilter(b, -1, (r, r), normalize=True)
        weights[:, :, c] = (mean_a * guide + mean_b).clip(0, 1).astype(np.float32)

    # Step 2: Normalize weights (ensure sum=1 per pixel)
    w_sum = weights.sum(axis=2, keepdims=True) + 1e-8
    weights /= w_sum

    # Step 3: Weighted fusion
    fused = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(n_src):
        fused += weights[:, :, c:c+1] * srcs_f32[c]

    fused = fused.clip(0, 255).astype(np.uint8)

    # Step 4: Bilateral anti-aliasing
    fused = cv2.bilateralFilter(fused, d=bilateral_d,
                                sigmaColor=bilateral_sigma_color,
                                sigmaSpace=bilateral_sigma_space)
    return fused


def edge_aware_feather(dec_seg, guide_img, feather_width=20, downsample=2):
    """
    边缘感知羽化: 降采样计算 alpha, 再 upsample 回原图.
    downsample=2 时计算量降为 1/4, 视觉无差异（alpha 本身就是高斯模糊的）.
    """
    h, w = dec_seg.shape
    ds = downsample
    dh, dw = h // ds, w // ds

    # 降采样决策图和引导图
    dec_small = cv2.resize(dec_seg.astype(np.float32), (dw, dh),
                           interpolation=cv2.INTER_NEAREST).astype(dec_seg.dtype)
    guide_gray_small = cv2.cvtColor(
        cv2.resize(guide_img, (dw, dh)), cv2.COLOR_RGB2GRAY)
    n_src = dec_small.max() + 1

    # 决策边界
    boundary = np.zeros((dh, dw), dtype=bool)
    for c in range(n_src):
        boundary |= cv2.Canny((dec_small == c).astype(np.uint8), 0, 1).astype(bool)

    # 图像边缘强度
    img_edge = np.abs(cv2.Sobel(guide_gray_small, cv2.CV_64F, 1, 1, ksize=3))
    img_edge /= (img_edge.max() + 1e-8)

    fw = max(1, feather_width // ds)  # 羽化宽度也要按比例缩小
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fw * 2 + 1, fw * 2 + 1))
    transition = cv2.dilate(boundary.astype(np.uint8), kernel)
    alpha_small = np.zeros((dh, dw), dtype=np.float32)
    alpha_small[transition > 0] = 1.0
    alpha_small *= (1.0 - img_edge)
    alpha_small = cv2.GaussianBlur(alpha_small, (fw * 2 + 1, fw * 2 + 1), fw / 2)

    # Upsample 回原图
    alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0, 1)


def fuse_quality(dec_seg, alpha, srcs, seg, seg_votes, bilateral_half_res=True,
                 bilateral_d=7, bilateral_sigma_color=30, bilateral_sigma_space=30):
    """
    融合: 硬融合 + 向量化软融合 + 羽化混合 + 双边滤波.
    与 infer_v5_full.py 的 fuse() 数学等价.
    """
    h, w = int(dec_seg.shape[0]), int(dec_seg.shape[1])
    n_src = len(srcs)
    srcs_f32 = [s.astype(np.float32) for s in srcs]

    # 1. 硬融合 — 与原始等价
    fused_hard = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(n_src):
        fused_hard += (dec_seg == c)[:, :, None] * srcs_f32[c]

    # 2. 软融合 — 向量化, 与原始逐SP循环数学等价
    # 原始: top2 = argsort(votes)[-2:] → top2[0]=第二大, top2[1]=最大
    # 本版: top2_asc[:,0]=第二大, top2_asc[:,1]=最大 (与原始完全一致)
    top2_asc = np.argsort(seg_votes, axis=1)[:, -2:]  # (n_sp, 2), ascending → last 2
    top2_counts = np.take_along_axis(seg_votes, top2_asc, axis=1)
    total = top2_counts.sum(axis=1, keepdims=True) + 1e-8
    sp_weights = top2_counts / total  # (n_sp, 2)

    fused_soft = np.zeros((h, w, 3), dtype=np.float32)
    seg_flat = seg.ravel()
    for c in range(n_src):
        # c是第二大源图? (top2_asc[:, 0])
        mask1 = (top2_asc[:, 0][seg] == c)
        if mask1.any():
            fused_soft[mask1] += sp_weights[:, 0][seg][mask1, None] * srcs_f32[c][mask1]
        # c是最大源图? (top2_asc[:, 1])
        mask2 = (top2_asc[:, 1][seg] == c)
        if mask2.any():
            fused_soft[mask2] += sp_weights[:, 1][seg][mask2, None] * srcs_f32[c][mask2]

    # 3. 羽化混合
    a = alpha[:, :, None]
    fused = (fused_hard * (1 - a) + fused_soft * a).astype(np.uint8)

    # 4. 双边抗锯齿
    if bilateral_half_res:
        dh, dw = h // 2, w // 2
        fused_small = cv2.resize(fused, (dw, dh))
        fused_small = cv2.bilateralFilter(fused_small, d=bilateral_d,
                sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
        fused = cv2.resize(fused_small, (w, h))
    else:
        fused = cv2.bilateralFilter(fused, d=bilateral_d,
                sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)

    return fused


# ============================================================
# 主流程
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='V5 答辩实时演示')
    p.add_argument('--group', default=None, help='测试集 group 名 (如 group_003)')
    p.add_argument('--data', default=str(ROOT / 'all_data' / 'split_data' / 'test'),
                   help='数据目录')
    p.add_argument('--imgs', nargs='*', default=None, help='自定义图片路径 (需5张)')
    p.add_argument('--ckpt', default=DEFAULT_CKPT, help='模型 checkpoint')
    p.add_argument('--mode', default='quality', choices=['fast', 'quality'],
                   help='fast=512x512单次推理 | quality=全分辨率分块推理')
    p.add_argument('--output', default=str(ROOT / 'output' / 'demo_realtime_fused.png'))
    p.add_argument('--tile', type=int, default=512)
    p.add_argument('--tile-batch', type=int, default=8,
                   help='分块推理batch大小 (越大越快但占更多显存)')
    p.add_argument('--felz-scale', type=int, default=200)
    p.add_argument('--felz-sigma', type=float, default=0.8)
    p.add_argument('--felz-min-size', type=int, default=200)
    p.add_argument('--feather-width', type=int, default=20,
                   help='羽化宽度 (fast模式建议5-10)')
    p.add_argument('--guide-idx', type=int, default=2, help='超像素引导图用第几张(0-based)')
    p.add_argument('--no-sp', action='store_true',
                   help='跳过超像素投票, 用中值滤波去噪 (更快, 质量接近)')
    p.add_argument('--seg-method', default='felz', choices=['felz', 'slic'],
                   help='超像素方法: felz=Felzenszwalb | slic=SLIC (更快)')
    p.add_argument('--bilateral-d', type=int, default=7,
                   help='双边滤波直径 (默认7, 越大抗锯齿越强)')
    p.add_argument('--bilateral-sigma-color', type=float, default=30)
    p.add_argument('--bilateral-sigma-space', type=float, default=30)
    p.add_argument('--smooth-decision', type=int, default=0,
                   help='决策边界平滑核大小 (0=禁用, 建议7-9, 保边消除锯齿)')
    p.add_argument('--merge-tiny-sp', type=int, default=0,
                   help='合并小于此像素数的碎片SP (0=禁用, 建议500-2000)')
    p.add_argument('--fusion-mode', default='standard', choices=['standard', 'guided'],
                   help='standard=硬融合+软融合+羽化 | guided=引导滤波直接融合(保边消锯齿)')
    p.add_argument('--guided-radius', type=int, default=15,
                   help='引导滤波半径 (仅guided模式, 建议10-30)')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'{"="*60}')
    print(f'V5 Multi-Focus Fusion — Live Demo')
    print(f'  Mode: {args.mode} | Device: {DEVICE} | Tile Batch: {args.tile_batch}')
    print(f'{"="*60}')

    # ================================================================
    # 1. Load model
    # ================================================================
    t_start = time.perf_counter()
    model = load_model(args.ckpt)
    print(f'\n[1/5] Model loaded (1.63M params)')

    # ================================================================
    # 2. Load sources
    # ================================================================
    if args.imgs:
        srcs = load_sources_from_files(args.imgs)
    elif args.group:
        srcs = load_sources_from_group(args.data, args.group)
    else:
        groups = sorted([d.name for d in Path(args.data).iterdir()
                        if d.is_dir() and d.name.startswith('group_')])
        args.group = groups[0]
        print(f'  No group specified, using: {args.group}')
        srcs = load_sources_from_group(args.data, args.group)

    H_orig, W_orig = srcs[0].shape[:2]
    n_src = len(srcs)
    guide = srcs[min(args.guide_idx, n_src - 1)]
    print(f'[2/5] Sources: {n_src} images, {W_orig}x{H_orig} | Guide: src[{args.guide_idx}]')

    # ================================================================
    # 3. Inference
    # ================================================================
    t_infer_start = time.perf_counter()
    t_fwd_total = 0.0

    if args.mode == 'fast':
        ratio = 512 / max(H_orig, W_orig)
        new_h, new_w = int(H_orig * ratio), int(W_orig * ratio)
        srcs_padded = []
        for s in srcs:
            rs = cv2.resize(s, (new_w, new_h))
            pad_h = (512 - new_h) // 2
            pad_w = (512 - new_w) // 2
            rs = cv2.copyMakeBorder(rs, pad_h, 512 - new_h - pad_h,
                                   pad_w, 512 - new_w - pad_w, cv2.BORDER_REFLECT)
            srcs_padded.append(rs)
        guide = srcs_padded[min(args.guide_idx, n_src - 1)]

        tiles = [torch.from_numpy(s.astype(np.float32) / 255.0).permute(2, 0, 1)
                 .unsqueeze(0).to(DEVICE) for s in srcs_padded]
        t0 = time.perf_counter()
        with torch.no_grad():
            dm = model(tiles)[1]
            dec_raw = dm.squeeze(2).argmax(dim=1)[0].cpu().numpy()
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t_fwd_total = (time.perf_counter() - t0) * 1000
        n_tiles = 1
        srcs_work = srcs_padded

    else:
        dec_raw, t_fwd_total, n_tiles = tiled_inference_batched(
            model, srcs, args.tile, args.tile_batch)

        srcs_work = srcs

    t_infer = time.perf_counter() - t_infer_start
    avg_fwd_ms = t_fwd_total / max(n_tiles, 1)
    n_batches = (n_tiles + args.tile_batch - 1) // args.tile_batch if args.mode == 'quality' else 1
    print(f'[3/5] Inference: {t_infer:.1f}s '
          f'({n_tiles} tiles x {avg_fwd_ms:.0f}ms avg, {n_batches} batches)')

    # ================================================================
    # 4. Superpixel voting
    # ================================================================
    t_post_start = time.perf_counter()

    fw = args.feather_width if args.mode == 'quality' else min(args.feather_width, 10)
    feather_ds = 1 if args.mode == 'quality' else 2
    use_half_bilateral = (args.mode == 'fast')
    t0 = time.perf_counter()
    dec_seg, seg, n_sp, seg_votes = segment_and_vote(
        dec_raw, guide, args.felz_scale, args.felz_sigma, args.felz_min_size)
    t_felz = time.perf_counter() - t0

    t_seg = time.perf_counter() - t_post_start
    print(f'[4/5] Superpixel voting: {t_seg:.1f}s '
          f'(Felz={t_felz:.1f}s, {n_sp} SPs, vectorized)')

    # ================================================================
    # 4b. Merge tiny SPs (optional)
    # ================================================================
    if args.merge_tiny_sp > 0:
        t_merge_start = time.perf_counter()
        dec_seg, n_merged = merge_tiny_sp(dec_seg, seg, seg_votes, args.merge_tiny_sp)
        t_merge = time.perf_counter() - t_merge_start
        print(f'[4b] Merge tiny SPs (<{args.merge_tiny_sp}px): {n_merged} merged ({t_merge:.1f}s)')

    # ================================================================
    # 4c. Decision smoothing (optional)
    # ================================================================
    if args.smooth_decision > 0:
        t_smooth_start = time.perf_counter()
        dec_seg = smooth_decision_boundary(dec_seg, guide, args.smooth_decision)
        t_smooth = time.perf_counter() - t_smooth_start
        print(f'[4c] Decision smoothing: {t_smooth:.1f}s (kernel={args.smooth_decision})')

    # ================================================================
    # 5. Fusion
    # ================================================================
    t_fuse_start = time.perf_counter()

    if args.fusion_mode == 'guided':
        fused = fuse_guided(dec_seg, srcs_work, guide, radius=args.guided_radius,
                            bilateral_d=args.bilateral_d,
                            bilateral_sigma_color=args.bilateral_sigma_color,
                            bilateral_sigma_space=args.bilateral_sigma_space)
        t_fuse = time.perf_counter() - t_fuse_start
        print(f'[5/5] Guided fusion: {t_fuse:.1f}s (r={args.guided_radius}, d={args.bilateral_d})')
    else:
        t0 = time.perf_counter()
        alpha = edge_aware_feather(dec_seg, guide, fw, downsample=feather_ds)
        t_feather = time.perf_counter() - t0

        t0 = time.perf_counter()
        fused = fuse_quality(dec_seg, alpha, srcs_work, seg, seg_votes,
                             bilateral_half_res=use_half_bilateral,
                             bilateral_d=args.bilateral_d,
                             bilateral_sigma_color=args.bilateral_sigma_color,
                             bilateral_sigma_space=args.bilateral_sigma_space)
        t_fuse_only = time.perf_counter() - t0

        t_fuse = time.perf_counter() - t_fuse_start
        print(f'[5/5] Feather+fuse+bilateral: {t_fuse:.1f}s '
              f'(feather={t_feather:.1f}s, fuse+bilateral={t_fuse_only:.1f}s, width={fw}px)')

    if args.mode == 'fast':
        pad_h = (512 - new_h) // 2
        pad_w = (512 - new_w) // 2
        fused = fused[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

    # ================================================================
    # Summary
    # ================================================================
    t_total = time.perf_counter() - t_start
    print(f'\n{"="*60}')
    print(f'Total: {t_total:.1f}s')
    print(f'  Inference:    {t_infer:.1f}s ({100*t_infer/t_total:.0f}%)')
    print(f'  SP voting:    {t_seg:.1f}s ({100*t_seg/t_total:.0f}%)')
    print(f'  Feather+fuse: {t_fuse:.1f}s ({100*t_fuse/t_total:.0f}%)')
    print(f'Output: {args.output}')
    print(f'{"="*60}')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
