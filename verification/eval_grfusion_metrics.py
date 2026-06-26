"""
评估 GRFusion 融合图 → 四指标（SF, AG, MI, QABF）
=====================================================

用途: 对 tools/GRFusion/fuse_5source.py 生成的融合图补全指标评估
评估协议: 与 V5 消融/IFCNN 相同的 MultiFocusDataset 协议
  - [0,1] 浮点值域
  - 512×512 评估尺寸
  - qabf 按 C(5,2)=10 对 pairwise 取均值

数据来源: output/grfusion_results/group_*_fused.png (10组, 5440×3648)
         → resize 到 512×512 评估
         → 对照源图: all_data/split_data/test/{group}/img_1~5.png

运行: python verification/eval_grfusion_metrics.py
输出: output/grfusion_metrics.json

创建日期: 2026-05-29
"""
import json, cv2, numpy as np
from pathlib import Path
from utils.metrics import spatial_frequency, average_gradient, mutual_information, qabf

BASE = Path(__file__).resolve().parent.parent
out_dir = BASE / 'output' / 'grfusion_results'
test_dir = BASE / 'all_data' / 'split_data' / 'test'

groups = sorted([d.name for d in test_dir.iterdir() if d.name.startswith('group_')])

all_sf, all_ag, all_mi, all_q = [], [], [], []
per_group = {}

for g in groups:
    fused_path = out_dir / f'{g}_fused.png'
    fused = cv2.imread(str(fused_path))
    fused_rgb = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    fused_gray = fused_rgb.mean(axis=2)

    srcs_gray = []
    for i in range(1, 6):
        s = cv2.imread(str(test_dir / g / f'img_{i}.png'))
        s_rgb = cv2.cvtColor(s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        s_g = cv2.resize(s_rgb.mean(axis=2), (512, 512))
        srcs_gray.append(s_g)

    sf = spatial_frequency(fused_gray)
    ag = average_gradient(fused_gray)
    mi = mutual_information(fused_gray, srcs_gray)
    qs = [qabf(fused_gray, srcs_gray[a], srcs_gray[b]) for a in range(5) for b in range(a+1, 5)]
    q = float(np.mean(qs))
    score = float(sf + 0.5 * ag)

    print(f'{g}: SF={sf:.4f} AG={ag:.4f} MI={mi:.4f} QABF={q:.4f} Score={score:.4f}')
    all_sf.append(sf); all_ag.append(ag); all_mi.append(mi); all_q.append(q)
    per_group[g] = {'SF': sf, 'AG': ag, 'MI': mi, 'QABF': q, 'Score': score}

mean = {
    'SF': round(float(np.mean(all_sf)), 4),
    'AG': round(float(np.mean(all_ag)), 4),
    'MI': round(float(np.mean(all_mi)), 4),
    'QABF': round(float(np.mean(all_q)), 4),
    'Score': round(float(np.mean(all_sf) + 0.5 * np.mean(all_ag)), 4),
}

print(f'\n=== GRFusion 10-group mean ===')
for k, v in mean.items():
    print(f'{k}: {v:.4f}')

out_path = BASE / 'output' / 'grfusion_metrics.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'method': 'GRFusion',
        'script': 'tools/GRFusion/fuse_5source.py',
        'eval_script': __file__,
        'eval_date': '2026-05-29',
        'protocol': 'MultiFocusDataset [0,1] 512x512 + qabf pairwise C(5,2)=10 mean',
        'groups': groups,
        'mean_metrics': mean,
        'per_group_metrics': per_group,
    }, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out_path}')
