"""论文图4-3: V5 Minimal vs m-SegNet V2 (梯度选源) 融合对比
用法: python verification/fig4_3_comparison.py
输出: output/figures/fig4_3_group_003.png, fig4_3_group_017.png
"""
import sys, cv2, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from models.m_segnet_v5_minimal import create_model
from demo_realtime import (
    load_sources_from_group, tiled_inference_batched,
    segment_and_vote, fuse_guided,
)

# ─── Load V5 Minimal ───
print('Loading V5 Minimal...')
ckpt = torch.load(
    str(BASE / 'runs/train/auto_05_v5_raw_feature/checkpoints/best.pt'),
    map_location='cpu', weights_only=False)
dn_weights = {k: v for k, v in ckpt['model'].items() if 'decision_net' in k}
model = create_model()
model.load_state_dict(dn_weights, strict=False)
model.to(DEVICE).eval()
print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

# ─── Config ───
TEST_DIR = str(BASE / 'all_data' / 'split_data' / 'test')
V2_DIR = Path('F:/Commercial project/multi-focus-pdf-depend/fused_results_v2')
OUT_DIR = BASE / 'output' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ['group_003', 'group_017']
ROI_SPECS = {
    'group_003': {'cx': 900, 'cy': 2820, 'R': 180, 'label': 'Dense mixed algae'},
    'group_017': {'cx': 1200, 'cy': 2000, 'R': 180, 'label': 'Filamentous algae'},
}


def generate_v5_fused(group_name):
    """V5 Minimal full pipeline inference"""
    srcs = load_sources_from_group(TEST_DIR, group_name)
    guide = srcs[2]

    # Tiled inference
    dec_raw, _, _ = tiled_inference_batched(model, srcs, 512, 8)

    # SP voting
    dec_seg, seg, n_sp, _ = segment_and_vote(dec_raw, guide)

    # Guided fusion
    fused = fuse_guided(dec_seg, srcs, guide, radius=25,
                        bilateral_d=5, bilateral_sigma_color=30, bilateral_sigma_space=30)

    return srcs, fused, dec_seg


for group_name in GROUPS:
    print(f'\n{"="*50}')
    print(f'  {group_name}')
    print(f'{"="*50}')

    # V5 inference
    srcs_v5, fused_v5, dec_v5 = generate_v5_fused(group_name)

    # Load V2 fused image
    v2_path = V2_DIR / f'{group_name}_fused.png'
    fused_v2 = cv2.cvtColor(cv2.imread(str(v2_path)), cv2.COLOR_BGR2RGB)

    # Ensure same dimensions
    h, w = fused_v5.shape[:2]
    fused_v2 = cv2.resize(fused_v2, (w, h))

    # ─── Build comparison figure ───
    roi = ROI_SPECS[group_name]
    cx, cy, R = roi['cx'], roi['cy'], roi['R']
    yr0, yr1 = max(0, cy - R), min(h, cy + R)
    xr0, xr1 = max(0, cx - R), min(w, cx + R)

    z = 2.0  # zoom factor
    crop_w, crop_h = xr1 - xr0, yr1 - yr0
    nw, nh = int(crop_w * z), int(crop_h * z)

    # V2 ROI
    v2_crop = cv2.resize(fused_v2[yr0:yr1, xr0:xr1], (nw, nh))
    # V5 ROI
    v5_crop = cv2.resize(fused_v5[yr0:yr1, xr0:xr1], (nw, nh))
    # Source 3 ROI (reference)
    s3_crop = cv2.resize(srcs_v5[2][yr0:yr1, xr0:xr1], (nw, nh))

    # Edge maps for ROI
    def sobel_edge(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        return (np.clip(np.sqrt(gx**2 + gy**2) * 3, 0, 255)).astype(np.uint8)

    v2_edge = cv2.cvtColor(sobel_edge(v2_crop), cv2.COLOR_GRAY2RGB)
    v5_edge = cv2.cvtColor(sobel_edge(v5_crop), cv2.COLOR_GRAY2RGB)

    # Label each panel — 左下角, 字体放大2倍
    def label(img, text, color=(0, 255, 0)):
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.1
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(img, text, (10, h - 12), font, scale, color, thickness)
        return img

    # Row 1: Source3(Ref) | V2 Fused | V5 Fused
    row1 = np.hstack([
        label(s3_crop.copy(), 'Source 3 (Ref)', (255,255,255)),
        label(v2_crop.copy(), 'm-SegNet V2', (255,80,80)),
        label(v5_crop.copy(), 'V5 Ours', (80,255,80)),
    ])

    # Row 2: V2 Edge | V5 Edge | Decision Map
    # Decision map ROI (colorize)
    dec_roi = dec_v5[yr0:yr1, xr0:xr1]
    dec_color = cv2.applyColorMap(
        cv2.resize(((dec_roi * 51) % 256).astype(np.uint8), (nw, nh)),
        cv2.COLORMAP_JET)
    dec_color = cv2.cvtColor(dec_color, cv2.COLOR_BGR2RGB)
    # Legend — match JET colormap colors (RGB order for RGB image)
    # JET(0)=#000080, JET(51)=#004CFF, JET(102)=#1AFFE6
    # JET(153)=#E6FF1A, JET(204)=#FF4C00
    jet_rgb = [(0,0,128),(0,76,255),(26,255,230),(230,255,26),(255,76,0)]
    names = ['S1','S2','S3','S4','S5']
    box_h = 18 + len(names) * 36
    box_w = 160
    overlay = dec_color.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (240, 240, 240), -1)
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (100, 100, 100), 2)
    dec_color = cv2.addWeighted(dec_color, 0.5, overlay, 0.5, 0)
    for i in range(5):
        r, g, b = jet_rgb[i]
        cv2.rectangle(dec_color, (12, 14 + i * 24), (32, 32 + i * 24), (r, g, b), -1)
        cv2.putText(dec_color, names[i], (42, 32 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

    row2 = np.hstack([
        label(v2_edge.copy(), 'V2 Edges', (255,80,80)),
        label(v5_edge.copy(), 'V5 Edges', (80,255,80)),
        label(dec_color.copy(), 'Decision Map', (255,255,255)),
    ])

    max_w = max(row1.shape[1], row2.shape[1])
    def padw(im, tw):
        if im.shape[1] >= tw: return im
        return np.hstack([im, np.zeros((im.shape[0], tw - im.shape[1], 3), dtype=np.uint8)])

    # Global context: small overview with ROI box
    overview_h = 200
    overview_scale = overview_h / h
    ov_w = int(w * overview_scale)
    v5_ov = cv2.resize(fused_v5, (ov_w, overview_h))
    v2_ov = cv2.resize(fused_v2, (ov_w, overview_h))
    # Draw ROI box
    def draw_roi_box(im, scale):
        cv2.rectangle(im,
                      (int(xr0 * scale), int(yr0 * scale)),
                      (int(xr1 * scale), int(yr1 * scale)),
                      (0, 255, 0), 2)
        return im
    v5_ov = draw_roi_box(v5_ov, overview_scale)
    v2_ov = draw_roi_box(v2_ov, overview_scale)
    label(v5_ov, 'V5 Full', (80,255,80))
    label(v2_ov, 'V2 Full', (255,80,80))
    overview = np.hstack([v2_ov, v5_ov])

    # Assemble final figure
    panel = np.vstack([padw(row1, max_w), padw(row2, max_w), padw(overview, max_w)])

    # Add title bar
    title_h = 40
    title = np.zeros((title_h, max_w, 3), dtype=np.uint8) + 40
    cv2.putText(title, f'{group_name} — {roi["label"]} | m-SegNet V2 vs V5 Ours (Gumbel hard decision + SP post-processing)',
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    panel = np.vstack([title, panel])

    out_path = OUT_DIR / f'fig4_3_{group_name}.png'
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f'  Saved: {out_path}  ({panel.shape[1]}x{panel.shape[0]})')

print(f'\nDone. Output: {OUT_DIR}/')
