"""
答辩一站式演示脚本
用法:
  python demo_defense.py                        # 默认 group_003
  python demo_defense.py --group group_017      # 指定组
  python demo_defense.py --all                  # 全部10组 + 汇总

输出 (output/):
  defense_fused.png       — 全分辨率融合图
  defense_panel.png       — 5源图+融合图+决策图 合成面板
  defense_zoom.png        — 细胞ROI硬决策放大
  defense_metrics.png     — 指标柱状图(vs IFCNN)
  defense_all_metrics.csv — 全部10组指标汇总 (--all模式)
"""
import argparse, sys, time, os
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from demo_realtime import (
    load_model, load_sources_from_group, tiled_inference_batched,
    segment_and_vote, fuse_guided, edge_aware_feather,
)
from utils.metrics import calculate_metrics, get_Qabf


def build_source_panel(srcs, fused, dec_seg, outpath):
    """5源图 + 融合图 + 决策图 合成面板"""
    h, w = srcs[0].shape[:2]
    # Each panel ~350px wide, 3 per row for sources, 2 for fused+decision
    nw = 350
    nh = int(h * nw / w)

    # Row 1: src1, src2, src3
    # Row 2: src4, src5, Fused
    # Row 3: Decision map, (blank), (blank)
    row1_panels = []
    for i in range(3):
        s = cv2.resize(srcs[i], (nw, nh))
        cv2.putText(s, f'Source {i+1}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        row1_panels.append(s)

    row2_panels = []
    for i in range(3, 5):
        s = cv2.resize(srcs[i], (nw, nh))
        cv2.putText(s, f'Source {i+1}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        row2_panels.append(s)
    f = cv2.resize(fused, (nw, nh))
    cv2.putText(f, 'FUSED RESULT', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    row2_panels.append(f)

    d = cv2.applyColorMap(cv2.resize(((dec_seg * 51) % 256).astype(np.uint8), (nw, nh)), cv2.COLORMAP_JET)
    cv2.putText(d, 'Decision Map', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Legend for decision map
    for i in range(5):
        cv2.putText(d, f'src{i+1}', (5, 50 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i], 1)

    row1 = np.hstack(row1_panels)
    row2 = np.hstack(row2_panels)
    row3 = np.hstack([d, np.zeros_like(d), np.zeros_like(d)])
    max_w = max(row1.shape[1], row2.shape[1], row3.shape[1])

    def padw(im, tw):
        if im.shape[1] >= tw: return im
        return np.hstack([im, np.zeros((im.shape[0], tw - im.shape[1], 3), dtype=np.uint8)])

    panel = np.vstack([padw(row1, max_w), padw(row2, max_w), padw(row3, max_w)])
    cv2.imwrite(outpath, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return panel


def build_zoom_panel(srcs, fused, dec_seg, seg, guide, outpath, cx=900, cy=2820, R=200):
    """细胞ROI放大: 5源图 + 融合图 + SP边界 + 羽化alpha"""
    yr0, yr1 = max(0, cy - R), min(fused.shape[0], cy + R)
    xr0, xr1 = max(0, cx - R), min(fused.shape[1], cx + R)

    alpha = edge_aware_feather(dec_seg, guide, 20, downsample=1)
    z = 1.5
    nw, nh = int((xr1 - xr0) * z), int((yr1 - yr0) * z)

    crops = []
    for i in range(5):
        c = cv2.resize(srcs[i][yr0:yr1, xr0:xr1], (nw, nh))
        cv2.putText(c, f'src{i+1}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        crops.append(c)

    f = cv2.resize(fused[yr0:yr1, xr0:xr1], (nw, nh))
    cv2.putText(f, 'Fused', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    crops.append(f)

    # SP boundary overlay
    sp_crop = seg[yr0:yr1, xr0:xr1]
    sp_b = np.zeros_like(sp_crop, dtype=bool)
    sp_b[:, 1:] |= (sp_crop[:, 1:] != sp_crop[:, :-1])
    sp_b[1:, :] |= (sp_crop[1:, :] != sp_crop[:-1, :])
    sp_ov = cv2.resize(fused[yr0:yr1, xr0:xr1], (nw, nh)).copy()
    sp_b_zoom = cv2.resize((sp_b * 255).astype(np.uint8), (nw, nh))
    sp_ov[sp_b_zoom > 128] = [0, 255, 0]
    cv2.putText(sp_ov, 'SP Boundaries', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    crops.append(sp_ov)

    # Alpha
    a = cv2.applyColorMap(cv2.resize((alpha[yr0:yr1, xr0:xr1] * 255).astype(np.uint8), (nw, nh)), cv2.COLORMAP_HOT)
    cv2.putText(a, 'Feather Alpha', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    crops.append(a)

    n_cols = 4
    rows = []
    for i in range(0, len(crops), n_cols):
        row = crops[i:i + n_cols]
        while len(row) < n_cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    panel = np.vstack(rows)
    cv2.imwrite(outpath, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return panel


def build_metrics_chart(metrics, ifcnn_metrics, outpath):
    """指标柱状图: V5 vs IFCNN"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Panel A: SF & AG
    ax = axes[0]
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w / 2, [metrics.get('spatial_frequency', 0), metrics.get('average_gradient', 0)],
           w, label='V5 (Ours)', color='#2196F3')
    ax.bar(x + w / 2, [ifcnn_metrics.get('SF', 0), ifcnn_metrics.get('AG', 0)],
           w, label='IFCNN', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(['SF', 'AG'])
    ax.set_title('Sharpness')
    ax.legend(fontsize=8)

    # Panel B: MI & EN
    ax = axes[1]
    x = np.arange(2)
    ax.bar(x - w / 2, [metrics.get('mutual_information', 0), metrics.get('entropy', 0)],
           w, label='V5 (Ours)', color='#4CAF50')
    ax.bar(x + w / 2, [ifcnn_metrics.get('MI', 0), 7.12],
           w, label='IFCNN', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(['MI', 'EN'])
    ax.set_title('Information')
    ax.legend(fontsize=8)

    # Panel C: QABF
    ax = axes[2]
    x = np.arange(1)
    ax.bar(x - w / 3, [metrics.get('qabf', 0)], w * 0.6, label='V5 (Ours)', color='#E91E63')
    ax.bar(x + w / 3, [ifcnn_metrics.get('QABF', 0)], w * 0.6, label='IFCNN', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(['QABF'])
    ax.set_title('Edge Preservation')
    ax.legend(fontsize=8)

    # Panel D: Score
    ax = axes[3]
    x = np.arange(1)
    ax.bar(x - w / 3, [metrics.get('spatial_frequency', 0) + 0.5 * metrics.get('average_gradient', 0)],
           w * 0.6, label='V5 (Ours)', color='#2196F3')
    ax.bar(x + w / 3, [ifcnn_metrics.get('Score', 0)], w * 0.6, label='IFCNN', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(['Score'])
    ax.set_title('Score (SF + 0.5*AG)')
    ax.legend(fontsize=8)

    fig.suptitle('V5 @ 5440x3648  |  IFCNN @ 512x512  |  SF/AG are resolution-dependent', fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def run_single(group, data_dir, model, ifcnn_metrics, output_dir, skip_metrics=False):
    """运行单个 group 的完整演示"""
    print(f'\n{"=" * 60}')
    print(f'  {group}')
    print(f'{"=" * 60}')

    t_start = time.perf_counter()

    # Load sources
    srcs = load_sources_from_group(data_dir, group)
    guide = srcs[2]
    h, w = srcs[0].shape[:2]
    print(f'  Sources: {len(srcs)} images, {w}x{h}')

    # Inference
    t0 = time.perf_counter()
    dec_raw, _, n_tiles = tiled_inference_batched(model, srcs, 512, 8)
    t_infer = time.perf_counter() - t0
    print(f'  Inference: {t_infer:.1f}s ({n_tiles} tiles)')

    # SP voting
    t0 = time.perf_counter()
    dec_seg, seg, n_sp, seg_votes = segment_and_vote(dec_raw, guide)
    t_vote = time.perf_counter() - t0
    print(f'  SP voting: {t_vote:.1f}s ({n_sp} SPs)')

    # Fusion (guided filter — edge-aware, smooths SP boundaries)
    t0 = time.perf_counter()
    fused = fuse_guided(dec_seg, srcs, guide, radius=25,
                        bilateral_d=5, bilateral_sigma_color=30, bilateral_sigma_space=30)
    t_fuse = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start
    print(f'  Fusion: {t_fuse:.1f}s | Total: {t_total:.1f}s')

    # Save fused image
    fused_path = output_dir / f'{group}_fused.png'
    cv2.imwrite(str(fused_path), cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))

    # Metrics: only 512x512 (matching paper protocol). Full-res skipped for speed.
    if skip_metrics:
        metrics_512 = {}
        score_512 = 0
    else:
        srcs_512 = [cv2.resize(s, (512, 512)) for s in srcs]
        tiles_512 = [torch.from_numpy(s.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE) for s in srcs_512]
        with torch.no_grad():
            fused_512_tensor = model(tiles_512)[0]
        fused_512_np = (fused_512_tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        fused_512_gray = fused_512_np.mean(axis=2).astype(np.float32) / 255.0
        srcs_512_gray = [s.mean(axis=2).astype(np.float32) / 255.0 for s in srcs_512]
        metrics_512 = calculate_metrics(fused_512_gray, srcs_512_gray)
        score_512 = metrics_512.get('spatial_frequency', 0) + 0.5 * metrics_512.get('average_gradient', 0)

    return {
        'group': group, 'time_total': t_total, 'time_infer': t_infer,
        'n_sp': n_sp, 'n_tiles': n_tiles,
        'sf': metrics_512.get('spatial_frequency', 0), 'ag': metrics_512.get('average_gradient', 0),
        'en': metrics_512.get('entropy', 0), 'mi': metrics_512.get('mutual_information', 0),
        'qabf': metrics_512.get('qabf', 0), 'score': score_512,
        'metrics_512': metrics_512, 'score_512': score_512,
        'srcs': srcs, 'fused': fused, 'dec_seg': dec_seg, 'seg': seg, 'guide': guide,
    }


def print_metrics_table(results, has_512=True):
    """打印指标表格 @ 512x512 (与IFCNN公平对比)"""
    if has_512 and results and results[0].get('score_512', 0) > 0:
        print(f"\n  === V5 @ 512x512 (matching paper protocol, fair comparison with IFCNN) ===")
        print(f"  {'Group':<12} {'SF':<8} {'AG':<8} {'EN':<8} {'MI':<8} {'QABF':<8} {'Score':<8} {'Time':<8}")
        print(f"  {'-'*84}")
        all_512 = []
        for r in results:
            s = r.get('score_512', 0)
            print(f"  {r['group']:<12} {r['sf']:<8.4f} {r['ag']:<8.4f} {r['en']:<8.2f} "
                  f"{r['mi']:<8.4f} {r['qabf']:<8.4f} {s:<8.4f} {r['time_total']:<8.1f}")
            all_512.append({'sf':r['sf'], 'ag':r['ag'], 'en':r['en'], 'mi':r['mi'],
                           'qabf':r['qabf'], 'score':s})
        if len(all_512) > 1:
            avg = {k: np.mean([x[k] for x in all_512]) for k in all_512[0]}
            print(f"  {'-'*84}")
            print(f"  {'AVERAGE':<12} {avg['sf']:<8.4f} {avg['ag']:<8.4f} {avg['en']:<8.2f} "
                  f"{avg['mi']:<8.4f} {avg['qabf']:<8.4f} {avg['score']:<8.4f} {'':<8}")


def main():
    p = argparse.ArgumentParser(description='V5 答辩一站式演示')
    p.add_argument('--group', default='group_003', help='测试组名')
    p.add_argument('--data', default=str(ROOT / 'all_data' / 'split_data' / 'test'))
    p.add_argument('--ckpt', default=str(ROOT / 'runs' / 'train' / 'auto_05_v5_raw_feature' / 'checkpoints' / 'best.pt'))
    p.add_argument('--output', default=str(ROOT / 'output'))
    p.add_argument('--all', action='store_true', help='运行全部10组')
    p.add_argument('--no-metrics', action='store_true', help='跳过指标评估, 仅生成图片(加速)')
    p.add_argument('--cx', type=int, default=900, help='放大ROI中心X')
    p.add_argument('--cy', type=int, default=2820, help='放大ROI中心Y')
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)

    # Load model (once)
    print('Loading model...')
    model = load_model(args.ckpt)
    print(f'V5_minimal model ready (27K params)')

    # Load IFCNN baseline
    ifcnn_path = ROOT / 'output' / 'ifcnn_metrics.json'
    ifcnn_metrics = {}
    if ifcnn_path.exists():
        import json
        with open(ifcnn_path) as f:
            ifcnn_metrics = json.load(f).get('mean_metrics', {})

    # Run
    if args.all:
        groups = sorted([d.name for d in data_dir.iterdir()
                        if d.is_dir() and d.name.startswith('group_')])
    else:
        groups = [args.group]

    results = []
    for group in groups:
        r = run_single(group, str(data_dir), model, ifcnn_metrics, output_dir, skip_metrics=args.no_metrics)
        results.append(r)

    # Print summary
    print_metrics_table(results, has_512=not args.no_metrics)

    # Save CSV (only when metrics are computed)
    if not args.no_metrics:
        csv_path = output_dir / 'defense_all_metrics.csv'
        with open(csv_path, 'w') as f:
            f.write('Group,SF,AG,EN,MI,QABF,Score,Time_s\n')
            for r in results:
                f.write(f"{r['group']},{r['sf']:.4f},{r['ag']:.4f},{r['en']:.2f},"
                        f"{r['mi']:.4f},{r['qabf']:.4f},{r['score']:.4f},{r['time_total']:.1f}\n")
        print(f'\nCSV saved: {csv_path}')

    # Generate visuals for the first/only group
    r = results[0]
    srcs, fused, dec_seg, seg, guide = r['srcs'], r['fused'], r['dec_seg'], r['seg'], r['guide']

    # Use 512x512 metrics for chart (fair SF/AG comparison with IFCNN)
    # MI/QABF/EN are less resolution-dependent
    metrics_chart = r.get('metrics_512', {'spatial_frequency': r['sf'], 'average_gradient': r['ag'],
                     'entropy': r['en'], 'mutual_information': r['mi'], 'qabf': r['qabf']})

    print(f'\n=== Generating visuals for {r["group"]} ===')

    # Panel: sources + fused + decision
    panel_path = output_dir / 'defense_panel.png'
    build_source_panel(srcs, fused, dec_seg, str(panel_path))
    print(f'  {panel_path}')

    # Zoom: ROI with SP boundaries
    zoom_path = output_dir / 'defense_zoom.png'
    build_zoom_panel(srcs, fused, dec_seg, seg, guide, str(zoom_path), args.cx, args.cy)
    print(f'  {zoom_path}')

    # V5 vs IFCNN comparison panel — with edge maps to prove SF advantage
    ifcnn_path = output_dir / 'ifcnn_group003.png'
    if ifcnn_path.exists():
        ifcnn_img = cv2.cvtColor(cv2.imread(str(ifcnn_path)), cv2.COLOR_BGR2RGB)
        v5_512_img = cv2.resize(fused, (512, 512))

        # Row 1: full images + edge maps
        def sobel_edge(img):
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
            gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
            e = np.sqrt(gx**2 + gy**2)
            return (np.clip(e * 3, 0, 255)).astype(np.uint8)  # amplify for visibility
        v5_edge = cv2.cvtColor(sobel_edge(v5_512_img), cv2.COLOR_GRAY2BGR)
        ifcnn_edge = cv2.cvtColor(sobel_edge(ifcnn_img), cv2.COLOR_GRAY2BGR)

        # Row 1: V5 | IFCNN | V5 Edge | IFCNN Edge
        row1 = np.hstack([v5_512_img, ifcnn_img, v5_edge, ifcnn_edge])
        cv2.putText(row1, 'V5 (Ours)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row1, 'IFCNN', (522, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row1, 'V5 Edges', (1034, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row1, 'IFCNN Edges', (1546, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Row 2: zoom ROI (center-left area with cell clusters)
        cx, cy, R = 200, 280, 120
        v5_zoom = v5_512_img[cy-R:cy+R, cx-R:cx+R]
        ifcnn_zoom = ifcnn_img[cy-R:cy+R, cx-R:cx+R]
        v5_zoom_e = v5_edge[cy-R:cy+R, cx-R:cx+R]
        ifcnn_zoom_e = ifcnn_edge[cy-R:cy+R, cx-R:cx+R]
        z = 2
        zh, zw = v5_zoom.shape[:2]
        row2 = np.hstack([cv2.resize(im, (zw*z, zh*z)) for im in [v5_zoom, ifcnn_zoom, v5_zoom_e, ifcnn_zoom_e]])
        cv2.putText(row2, f'Score={r["score_512"]:.3f}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row2, f'Score=0.216', (zw*z+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        sf_val = metrics_chart['spatial_frequency']
        cv2.putText(row2, f'SF={sf_val:.3f}', (zw*z*2+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row2, 'SF=0.089', (zw*z*3+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        def padw2(im, tw):
            if im.shape[1] >= tw: return im
            return np.hstack([im, np.zeros((im.shape[0], tw - im.shape[1], 3), dtype=np.uint8)])
        max_w = max(row1.shape[1], row2.shape[1])
        cmp_panel = np.vstack([padw2(row1, max_w), padw2(row2, max_w)])
        cmp_path = output_dir / 'defense_vs_ifcnn.png'
        cv2.imwrite(str(cmp_path), cv2.cvtColor(cmp_panel, cv2.COLOR_RGB2BGR))
        print(f'  {cmp_path}')
    build_zoom_panel(srcs, fused, dec_seg, seg, guide, str(zoom_path), args.cx, args.cy)
    print(f'  {zoom_path}')


    # Full fused (already saved as group_xxx_fused.png)
    fused_file = output_dir / f'{r["group"]}_fused.png'
    print(f'  {fused_file}')

    # Final summary
    print(f'\n{"=" * 60}')
    print(f'DEMO READY')
    print(f'{"=" * 60}')
    print(f'  Group: {r["group"]}')
    print(f'  Total time: {r["time_total"]:.1f}s')
    print(f'  SPs: {r["n_sp"]}')
    print(f'  MI: {r["mi"]:.4f} | QABF: {r["qabf"]:.4f} | Score: {r["score"]:.4f}')
    print(f'\n  Output files:')
    print(f'    {panel_path}')
    print(f'    {zoom_path}')
    fused_out = output_dir / f'{r["group"]}_fused.png'
    print(f'    {fused_out}')


if __name__ == '__main__':
    main()
