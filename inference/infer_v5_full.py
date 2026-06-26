"""
V5 完整推理管线 — 一键生成论文级融合结果

流程:
  1. V5 模型瓦片推理 → 逐像素决策
  2. Felzenszwalb 分割 → 有机区域
  3. 区域多数投票 → 区域一致性决策
  4. 边缘感知羽化 → 结构边硬切换 + 平坦区平滑过渡
  5. 双边抗锯齿 → 消除斜边阶梯感

用法:
  python infer_v5_full.py --group group_003
  python infer_v5_full.py --data all_data/split_data/test --output output/fused
"""
import sys, os, argparse
from pathlib import Path
import torch, cv2, numpy as np, pickle
from skimage.segmentation import felzenszwalb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.m_segnet_v5 import create_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_CKPT = ROOT / 'runs' / 'train' / 'auto_05_v5_raw_feature' / 'checkpoints' / 'best.pt'


def parse_args():
    p = argparse.ArgumentParser(description='V5 Full Pipeline Inference')
    p.add_argument('--group', default=None, help='单个 group 名 (如 group_003)')
    p.add_argument('--data', default=str(ROOT / 'all_data' / 'split_data' / 'test'), help='数据目录')
    p.add_argument('--ckpt', default=str(DEFAULT_CKPT), help='模型 checkpoint')
    p.add_argument('--output', default=str(ROOT / 'output' / 'fused'), help='输出目录')
    p.add_argument('--tile', type=int, default=512, help='瓦片大小')
    p.add_argument('--felz-scale', type=int, default=200, help='Felzenszwalb scale')
    p.add_argument('--felz-sigma', type=float, default=0.8, help='Felzenszwalb sigma')
    p.add_argument('--felz-min-size', type=int, default=200, help='Felzenszwalb min size')
    p.add_argument('--feather-width', type=int, default=20, help='羽化宽度 (px)')
    p.add_argument('--bilateral-d', type=int, default=5, help='Bilateral filter diameter')
    p.add_argument('--bilateral-sigma-color', type=float, default=30)
    p.add_argument('--bilateral-sigma-space', type=float, default=30)
    p.add_argument('--save-intermediates', action='store_true', help='保存中间结果')
    return p.parse_args()


def load_sources(data_dir, group):
    """加载5张源图"""
    srcs = []
    for i in range(1, 6):
        img = cv2.imread(str(Path(data_dir) / group / f'img_{i}.png'))
        if img is None:
            raise FileNotFoundError(f'{data_dir}/{group}/img_{i}.png')
        srcs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return srcs


def tiled_inference(model, srcs, tile_size=512):
    """瓦片推理 → 全局决策图"""
    h, w = srcs[0].shape[:2]
    ty = (h + tile_size - 1) // tile_size
    tx = (w + tile_size - 1) // tile_size
    global_dec = np.zeros((h, w), dtype=np.int32)

    for yi in range(ty):
        for xi in range(tx):
            y0 = yi * tile_size; y1 = min(y0 + tile_size, h)
            x0 = xi * tile_size; x1 = min(x0 + tile_size, w)
            tiles = []
            for src in srcs:
                t = cv2.resize(src[y0:y1, x0:x1], (512, 512))
                tiles.append(torch.from_numpy(t.astype(np.float32) / 255.0)
                            .permute(2, 0, 1).unsqueeze(0).to(DEVICE))
            with torch.no_grad():
                dm = model(tiles)[1]
                dec = dm.squeeze(2).argmax(dim=1)[0].cpu().numpy()
                dec_full = cv2.resize(dec.astype(np.float32), (x1 - x0, y1 - y0),
                                      interpolation=cv2.INTER_NEAREST)
                global_dec[y0:y1, x0:x1] = dec_full.astype(np.int32)
    return global_dec


def segment_and_vote(dec_map, guide_img, args):
    """Felzenszwalb 分割 + 区域内多数投票"""
    h, w = dec_map.shape
    # Downsample guide for speed
    guide_small = cv2.resize(guide_img, (w // 4, h // 4))
    seg_small = felzenszwalb(guide_small, scale=args.felz_scale,
                             sigma=args.felz_sigma, min_size=args.felz_min_size)
    seg = cv2.resize(seg_small.astype(np.float32), (w, h),
                     interpolation=cv2.INTER_NEAREST).astype(np.int32)

    n_sp = seg.max() + 1
    dec_seg = np.zeros((h, w), dtype=np.int32)
    for sp in range(n_sp):
        mask = seg == sp
        if mask.sum() == 0: continue
        votes = np.bincount(dec_map[mask], minlength=5)
        dec_seg[mask] = votes.argmax()

    return dec_seg, seg, n_sp


def edge_aware_feather(dec_seg, guide_img, args):
    """边缘感知羽化融合"""
    h, w = dec_seg.shape
    guide_gray = cv2.cvtColor(guide_img, cv2.COLOR_RGB2GRAY)

    # SP boundaries
    boundary = np.zeros((h, w), dtype=bool)
    for c in range(5):
        b = cv2.Canny((dec_seg == c).astype(np.uint8), 0, 1)
        boundary |= b.astype(bool)

    # Image edge strength
    img_edge = np.abs(cv2.Sobel(guide_gray, cv2.CV_64F, 1, 1, ksize=3))
    img_edge /= img_edge.max()

    # Feather alpha
    fw = args.feather_width
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fw * 2 + 1, fw * 2 + 1))
    transition = cv2.dilate(boundary.astype(np.uint8), kernel)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[transition > 0] = 1.0
    alpha *= (1.0 - img_edge)
    alpha = cv2.GaussianBlur(alpha, (fw * 2 + 1, fw * 2 + 1), fw / 2)

    return alpha


def fuse(dec_seg, alpha, srcs, seg, dec_raw):
    """硬融合 + 软融合 → 羽化混合 → 双边抗锯齿"""
    h, w = dec_seg.shape
    n_sp = seg.max() + 1

    # Hard fused
    fused_hard = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(5):
        fused_hard += (dec_seg == c)[:, :, None] * srcs[c].astype(np.float32)

    # Soft fused (top-2 blend per segment)
    fused_soft = np.zeros((h, w, 3), dtype=np.float32)
    for sp in range(n_sp):
        mask = seg == sp
        if mask.sum() == 0: continue
        votes = np.bincount(dec_raw[mask], minlength=5)
        top2 = np.argsort(votes)[-2:]
        total = votes[top2[0]] + votes[top2[1]] + 1e-8
        w1, w2 = votes[top2[0]] / total, votes[top2[1]] / total
        blended = srcs[top2[0]].astype(np.float32) * w1 + srcs[top2[1]].astype(np.float32) * w2
        fused_soft[mask] = blended[mask]

    # Blend
    a = alpha[:, :, None]
    fused = (fused_hard * (1 - a) + fused_soft * a).astype(np.uint8)

    # Bilateral anti-aliasing
    fused = cv2.bilateralFilter(fused, d=5, sigmaColor=30, sigmaSpace=30)

    return fused


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load model
    model = create_model(num_source_images=5, use_fusion_head='gumbel', top_k=1)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(DEVICE).eval()
    print(f'Model loaded: {args.ckpt}')

    # Find groups
    data_dir = Path(args.data)
    if args.group:
        groups = [args.group]
    else:
        groups = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('group_')])
    print(f'Processing {len(groups)} groups...')

    for group in groups:
        print(f'\n{"="*50}\n{group}\n{"="*50}')
        srcs = load_sources(data_dir, group)
        h, w = srcs[0].shape[:2]

        # Step 1: Tiled inference
        print('  [1/5] Tiled inference...')
        dec_raw = tiled_inference(model, srcs, args.tile)

        # Step 2: Segmentation + vote
        print('  [2/5] Felzenszwalb segmentation...')
        guide = srcs[2]  # Source 3 as guide
        dec_seg, seg, n_sp = segment_and_vote(dec_raw, guide, args)
        print(f'    {n_sp} segments')

        # Step 3: Edge-aware feather
        print('  [3/5] Edge-aware feathering...')
        alpha = edge_aware_feather(dec_seg, guide, args)

        # Step 4: Fuse + bilateral
        print('  [4/5] Fusing + bilateral...')
        fused = fuse(dec_seg, alpha, srcs, seg, dec_raw)

        # Step 5: Save
        print('  [5/5] Saving...')
        out_path = Path(args.output) / f'{group}_fused.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))
        print(f'  Saved: {out_path}')

        if args.save_intermediates:
            for name, data in [('decision_raw', dec_raw), ('decision_seg', dec_seg),
                               ('segments', seg), ('alpha', (alpha * 255).astype(np.uint8))]:
                cv2.imwrite(str(Path(args.output) / f'{group}_{name}.png'), data)

    print(f'\nDone. Output: {args.output}')


if __name__ == '__main__':
    main()
