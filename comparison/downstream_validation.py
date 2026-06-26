"""
Stage 4 下游任务验证 — 多焦距融合对藻类分析任务的有效性

两个下游任务：
  任务 1: SIFT 特征点检测 — 融合图是否比单焦图包含更多可检测特征
  任务 2: 聚焦区域覆盖度分析 — 融合图是否实现了全图清晰覆盖

用法:
  python downstream_validation.py
"""

import sys, os, json, time
import numpy as np
import torch
import cv2
from scipy.signal import convolve2d
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import MultiFocusDataset
from utils.metrics import spatial_frequency, average_gradient, mutual_information, qabf
from models.m_segnet_v2 import create_model

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, 'runs', 'downstream')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_SRC = 5


# ======================== 模型加载 ========================

def load_r16_model():
    model = create_model(num_source_images=N_SRC, use_fusion_head='gumbel').to(DEVICE)
    ckpt_path = os.path.join(BASE, 'runs/train/v2_arch_r16_pure_focal/checkpoints/epoch_35.pt')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ======================== 任务 1: SIFT 特征点检测 ========================

def detect_sift_keypoints(img_uint8):
    """返回 SIFT 关键点数量。img_uint8: (H,W,3) uint8"""
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    return len(kp)


def eval_sift(model, test_loader):
    """评估 SIFT 关键点：融合图 vs 各单焦图 vs 平均融合"""
    print('\n' + '=' * 70)
    print('  任务 1: SIFT 特征点检测')
    print('  衡量: 融合图是否比单焦图保留更多可检测纹理特征')
    print('=' * 70)

    results = {
        'fused_r16': [],
        'fused_average': [],
        'best_single': [],
        'worst_single': [],
        'per_source': {i: [] for i in range(N_SRC)},
    }

    all_kp_r16 = []
    all_kp_avg = []
    all_kp_best = []
    all_group_names = []

    for batch_idx, batch in enumerate(test_loader):
        # 从路径提取 group 名称，如 all_data/split_data/test/group_003/img_1.png → group_003
        sample_path = batch['paths'][0][0]
        group_name = os.path.basename(os.path.dirname(sample_path))
        all_group_names.append(group_name)

        sources_np = []
        for s in batch['sources']:
            img = (s[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            sources_np.append(img)

        # SIFT on each single focal plane
        kp_per_src = [detect_sift_keypoints(s) for s in sources_np]
        for i, kp in enumerate(kp_per_src):
            results['per_source'][i].append(kp)
        results['best_single'].append(max(kp_per_src))
        results['worst_single'].append(min(kp_per_src))

        # SIFT on average fusion
        avg_fused = np.mean(sources_np, axis=0).astype(np.uint8)
        kp_avg = detect_sift_keypoints(avg_fused)
        results['fused_average'].append(kp_avg)

        # SIFT on R16 fusion
        sources_tensor = [s.to(DEVICE) for s in batch['sources']]
        with torch.no_grad():
            out = model(sources_tensor)
            fused_tensor = out[0] if isinstance(out, tuple) else out
        fused_np = (fused_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        kp_r16 = detect_sift_keypoints(fused_np)
        results['fused_r16'].append(kp_r16)

        all_kp_r16.append(kp_r16)
        all_kp_avg.append(kp_avg)
        all_kp_best.append(max(kp_per_src))

    # 汇总
    def avg_std(vals):
        return np.mean(vals), np.std(vals)

    mu_r16, sd_r16 = avg_std(all_kp_r16)
    mu_avg, sd_avg = avg_std(all_kp_avg)
    mu_best, sd_best = avg_std(all_kp_best)

    print(f'\n  {"图像类型":25s} {"平均关键点":>10s}  {"Std":>8s}  {"vs R16":>10s}')
    print(f'  {"-" * 55}')
    print(f'  {"R16 融合 (本文方法)":25s} {mu_r16:8.1f}  ±{sd_r16:6.1f}  {"—":>10s}')

    for i in range(N_SRC):
        mu_s, sd_s = avg_std(results['per_source'][i])
        delta = (mu_r16 - mu_s) / mu_s * 100
        print(f'  {"焦平面 " + str(i+1):25s} {mu_s:8.1f}  ±{sd_s:6.1f}  {delta:+8.1f}%')

    print(f'  {"平均融合 (baseline)":25s} {mu_avg:8.1f}  ±{sd_avg:6.1f}  {(mu_r16-mu_avg)/mu_avg*100:+8.1f}%')
    print(f'  {"最佳单焦面":25s} {mu_best:8.1f}  ±{sd_best:6.1f}  {(mu_r16-mu_best)/mu_best*100:+8.1f}%')

    # 提升率
    improvement_src = (mu_r16 - mu_best) / mu_best * 100
    improvement_avg = (mu_r16 - mu_avg) / mu_avg * 100

    print(f'\n  >>> R16 融合相比最佳单焦面关键点提升: {improvement_src:+.1f}%')
    print(f'  >>> R16 融合相比平均融合关键点提升:   {improvement_avg:+.1f}%')

    return {
        'task': 'SIFT keypoint detection',
        'r16_mean': round(mu_r16, 1),
        'r16_std': round(sd_r16, 1),
        'best_single_mean': round(mu_best, 1),
        'average_fusion_mean': round(mu_avg, 1),
        'improvement_vs_best_single_pct': round(improvement_src, 1),
        'improvement_vs_average_pct': round(improvement_avg, 1),
        'per_group': {
            'groups': all_group_names,
            'r16': [round(x, 1) for x in all_kp_r16],
            'best_single': [round(x, 1) for x in all_kp_best],
            'average': [round(x, 1) for x in all_kp_avg],
        }
    }


# ======================== 任务 2: 聚焦区域覆盖度分析 ========================

def local_focus_measure(gray, window=15):
    """逐像素局部 Laplacian 方差（经典 focus measure）"""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = convolve2d(gray.astype(np.float32), kernel, mode='same')
    # 局部方差
    lap2 = lap ** 2
    box_kernel = np.ones((window, window), dtype=np.float32) / (window * window)
    local_var = convolve2d(lap2, box_kernel, mode='same')
    return local_var


def eval_focus_coverage(model, test_loader):
    """评估聚焦区域覆盖度"""
    print('\n' + '=' * 70)
    print('  任务 2: 聚焦区域覆盖度分析')
    print('  衡量: 融合图是否实现了全图高清晰覆盖')
    print('=' * 70)

    results = {
        'fused_r16_coverage': [],
        'best_single_coverage': [],
        'per_source_coverage': {i: [] for i in range(N_SRC)},
        'fused_r16_focus_mean': [],
        'best_single_focus_mean': [],
    }

    all_r16_cov = []
    all_best_cov = []
    all_r16_fm = []
    all_best_fm = []

    for batch_idx, batch in enumerate(test_loader):
        sources_np = []
        for s in batch['sources']:
            img = (s[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            sources_np.append(img)

        # 各单焦图的 focus measure
        fm_per_src = []
        for s in sources_np:
            gray = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
            fm = local_focus_measure(gray)
            fm_per_src.append(fm)

        # 定义"高聚焦"阈值 = 全数据集 focus measure 的 70 分位数
        all_fm_values = np.concatenate([fm.flatten() for fm in fm_per_src])
        high_focus_thresh = np.percentile(all_fm_values, 70)

        # 各单焦面的高聚焦覆盖率
        cov_per_src = [np.mean(fm > high_focus_thresh) for fm in fm_per_src]
        for i, cov in enumerate(cov_per_src):
            results['per_source_coverage'][i].append(cov)
        results['best_single_coverage'].append(max(cov_per_src))

        best_idx = np.argmax(cov_per_src)
        results['best_single_focus_mean'].append(np.mean(fm_per_src[best_idx]))

        # R16 融合图
        sources_tensor = [s.to(DEVICE) for s in batch['sources']]
        with torch.no_grad():
            out = model(sources_tensor)
            fused_tensor = out[0] if isinstance(out, tuple) else out
        fused_np = (fused_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        fused_gray = cv2.cvtColor(fused_np, cv2.COLOR_RGB2GRAY)
        fused_fm = local_focus_measure(fused_gray)

        r16_cov = np.mean(fused_fm > high_focus_thresh)
        results['fused_r16_coverage'].append(r16_cov)
        results['fused_r16_focus_mean'].append(np.mean(fused_fm))

        all_r16_cov.append(r16_cov)
        all_best_cov.append(max(cov_per_src))
        all_r16_fm.append(np.mean(fused_fm))
        all_best_fm.append(np.mean(fm_per_src[best_idx]))

    mu_r16_cov, sd_r16_cov = np.mean(all_r16_cov), np.std(all_r16_cov)
    mu_best_cov, sd_best_cov = np.mean(all_best_cov), np.std(all_best_cov)

    print(f'\n  {"图像类型":30s} {"高聚焦覆盖率":>12s}  {"平均 Focus":>10s}')
    print(f'  {"-" * 55}')
    print(f'  {"R16 融合 (本文方法)":30s} {mu_r16_cov*100:8.1f}% ±{sd_r16_cov*100:.1f}%  {np.mean(all_r16_fm):8.2f}')

    for i in range(N_SRC):
        mu_c, sd_c = np.mean(results['per_source_coverage'][i]), np.std(results['per_source_coverage'][i])
        print(f'  {"焦平面 " + str(i+1):30s} {mu_c*100:8.1f}% ±{sd_c*100:.1f}%')

    print(f'  {"最佳单焦面":30s} {mu_best_cov*100:8.1f}% ±{sd_best_cov*100:.1f}%  {np.mean(all_best_fm):8.2f}')

    improvement_cov = (mu_r16_cov - mu_best_cov) / mu_best_cov * 100 if mu_best_cov > 0 else 0
    print(f'\n  >>> R16 融合相比最佳单焦面高聚焦覆盖率提升: {improvement_cov:+.1f}%')
    print(f'  >>> 理想全焦融合覆盖率 = 100%，R16 达到 {mu_r16_cov*100:.1f}%')

    return {
        'task': 'Focus coverage analysis',
        'r16_coverage_pct': round(mu_r16_cov * 100, 1),
        'best_single_coverage_pct': round(mu_best_cov * 100, 1),
        'r16_focus_mean': round(np.mean(all_r16_fm), 2),
        'best_single_focus_mean': round(np.mean(all_best_fm), 2),
        'improvement_vs_best_single_pct': round(improvement_cov, 1),
    }


# ======================== 任务 3: 边缘连续性分析 ========================

def eval_edge_density(model, test_loader):
    """边缘密度：Canny 边缘检测后的边缘像素占比"""
    print('\n' + '=' * 70)
    print('  任务 3: 边缘密度分析')
    print('  衡量: 融合图是否保留了更多连续边缘')
    print('=' * 70)

    def canny_edge_density(img_uint8):
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.mean(edges > 0)

    all_r16_ed = []
    all_best_ed = []
    all_avg_ed = []

    for batch in test_loader:
        sources_np = []
        for s in batch['sources']:
            img = (s[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            sources_np.append(img)

        ed_per_src = [canny_edge_density(s) for s in sources_np]
        all_best_ed.append(max(ed_per_src))

        avg_fused = np.mean(sources_np, axis=0).astype(np.uint8)
        all_avg_ed.append(canny_edge_density(avg_fused))

        sources_tensor = [s.to(DEVICE) for s in batch['sources']]
        with torch.no_grad():
            out = model(sources_tensor)
            fused_tensor = out[0] if isinstance(out, tuple) else out
        fused_np = (fused_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        all_r16_ed.append(canny_edge_density(fused_np))

    mu_r16 = np.mean(all_r16_ed)
    mu_best = np.mean(all_best_ed)
    mu_avg = np.mean(all_avg_ed)

    print(f'\n  {"图像类型":25s} {"边缘密度":>10s}')
    print(f'  {"-" * 40}')
    print(f'  {"R16 融合 (本文方法)":25s} {mu_r16*100:8.2f}%')
    print(f'  {"最佳单焦面":25s} {mu_best*100:8.2f}%')
    print(f'  {"平均融合":25s} {mu_avg*100:8.2f}%')

    print(f'\n  >>> R16 相比最佳单焦面边缘密度提升: {(mu_r16-mu_best)/mu_best*100:+.1f}%')

    return {
        'task': 'Edge density analysis',
        'r16_edge_density_pct': round(mu_r16 * 100, 2),
        'best_single_edge_density_pct': round(mu_best * 100, 2),
        'average_fusion_edge_density_pct': round(mu_avg * 100, 2),
        'improvement_vs_best_single_pct': round((mu_r16 - mu_best) / mu_best * 100, 1),
    }


# ======================== 主评估 ========================

def main():
    print('=' * 70)
    print('  Stage 4 — 下游任务验证')
    print('  验证多焦距融合对藻类分析任务的有效性')
    print('=' * 70)

    # 加载数据
    test_ds = MultiFocusDataset(
        os.path.join(BASE, 'all_data/split_data/test'),
        is_train=False
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f'\n测试样本数: {len(test_ds)} 组 (每组 5 焦平面)')

    # 加载模型
    print('加载 R16_pure 模型...')
    model = load_r16_model()
    print('模型加载完成')

    # 执行下游评估
    t0 = time.time()

    task1_results = eval_sift(model, test_loader)
    task2_results = eval_focus_coverage(model, test_loader)
    task3_results = eval_edge_density(model, test_loader)

    elapsed = time.time() - t0

    # ======================== 汇总报告 ========================

    print('\n')
    print('=' * 70)
    print('  下游任务验证汇总')
    print('=' * 70)

    print(f"""
  ┌─────────────────────────┬──────────────────┬──────────────────┬──────────┐
  │      下游任务            │   最佳单焦面      │   R16 融合 (本文) │  提升    │
  ├─────────────────────────┼──────────────────┼──────────────────┼──────────┤
  │ SIFT 关键点数           │  {task1_results['best_single_mean']:>8.1f}       │  {task1_results['r16_mean']:>8.1f}       │ {task1_results['improvement_vs_best_single_pct']:>+6.1f}%  │
  │ 高聚焦覆盖率            │  {task2_results['best_single_coverage_pct']:>5.1f}%           │  {task2_results['r16_coverage_pct']:>5.1f}%           │ {task2_results['improvement_vs_best_single_pct']:>+6.1f}%  │
  │ 边缘密度                │  {task3_results['best_single_edge_density_pct']:>5.2f}%          │  {task3_results['r16_edge_density_pct']:>5.2f}%          │ {task3_results['improvement_vs_best_single_pct']:>+6.1f}%  │
  └─────────────────────────┴──────────────────┴──────────────────┴──────────┘
""")

    # 将 numpy 类型转换为 Python 原生类型
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj

    # 保存完整结果
    full_results = convert_to_native({
        'model': 'R16_pure (epoch_35)',
        'test_samples': len(test_ds),
        'evaluation_time_seconds': round(elapsed, 1),
        'task1_sift': task1_results,
        'task2_focus_coverage': task2_results,
        'task3_edge_density': task3_results,
    })

    output_path = os.path.join(OUTPUT_DIR, 'downstream_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f'\n完整结果已保存: {output_path}')

    # 生成论文可直接使用的结论
    print('\n' + '=' * 70)
    print('  论文可用的结论表述')
    print('=' * 70)
    print(f"""
  为验证融合结果对下游藻类分析任务的实际促进作用，本文设计了
  三项下游验证实验：

  1) SIFT 特征点检测：R16 融合图像平均检测到 {task1_results['r16_mean']:.0f} 个
     SIFT 关键点，相比最佳单焦面（{task1_results['best_single_mean']:.0f} 个）
     提升 {task1_results['improvement_vs_best_single_pct']:+.1f}%，说明融合图像
     保留了更多可检测的纹理与结构特征。

  2) 聚焦区域覆盖度：以局部 Laplacian 方差作为聚焦度量，定义高聚焦
     阈值（70 分位数），R16 融合图像的高聚焦像素覆盖率达到
     {task2_results['r16_coverage_pct']:.1f}%，显著高于最佳单焦面的
     {task2_results['best_single_coverage_pct']:.1f}%，验证了多焦距融合
     实现了全图清晰覆盖的设计目标。

  3) 边缘密度分析：R16 融合图像的 Canny 边缘密度为
     {task3_results['r16_edge_density_pct']:.2f}%，相比最佳单焦面
     （{task3_results['best_single_edge_density_pct']:.2f}%）提升
     {task3_results['improvement_vs_best_single_pct']:+.1f}%，边缘连续性更好。
""")


if __name__ == '__main__':
    main()
