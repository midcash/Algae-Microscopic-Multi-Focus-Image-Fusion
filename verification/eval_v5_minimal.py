"""V5 Minimal 评估 — 仅 DecisionNet(27K) + Gumbel 融合头
对照: V5 Full 0.2406±0.026 (1.63M)
用法: python verification/eval_v5_minimal.py
"""
import sys, torch, numpy as np, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, '.')
from models.m_segnet_v5_minimal import create_model, count_parameters
from utils.metrics import spatial_frequency, average_gradient, mutual_information, qabf
from utils.data_loader import MultiFocusDataset
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parent.parent
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_model(num_source_images=5)
n_params = count_parameters(model)
print(f'MSegNetV5Minimal: {n_params:,} params ({n_params/1e6:.4f}M)')
model.to(DEV).eval()

dataset = MultiFocusDataset(str(BASE / 'all_data' / 'split_data' / 'test'), is_train=False)
groups = [Path(g['images'][0]).parent.name for g in dataset.groups]

per_group = []
with torch.no_grad():
    for idx, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
        srcs = [s.to(DEV) for s in batch['sources']]
        fused = model(srcs)[0].cpu()
        fg = fused[0].numpy().mean(axis=0)
        sg = [s.cpu()[0].numpy().mean(axis=0) for s in batch['sources']]
        sf = float(spatial_frequency(fg))
        ag = float(average_gradient(fg))
        mi = float(mutual_information(fg, sg))
        q = float(np.mean([qabf(fg, sg[a], sg[b])
                           for a in range(5) for b in range(a + 1, 5)]))
        sc = round(sf + 0.5 * ag, 4)
        per_group.append({
            'group': groups[idx],
            'SF': round(sf, 4), 'AG': round(ag, 4), 'MI': round(mi, 4),
            'QABF': round(q, 4), 'Score': sc,
        })
        print(f'{groups[idx]}: SF={sf:.4f} AG={ag:.4f} MI={mi:.4f} QABF={q:.4f} Score={sc:.4f}')

all_sf = [g['SF'] for g in per_group]
all_ag = [g['AG'] for g in per_group]
all_mi = [g['MI'] for g in per_group]
all_q = [g['QABF'] for g in per_group]
all_sc = [g['Score'] for g in per_group]

mean = {
    'SF': round(np.mean(all_sf), 4), 'SF_std': round(np.std(all_sf, ddof=1), 4),
    'AG': round(np.mean(all_ag), 4), 'AG_std': round(np.std(all_ag, ddof=1), 4),
    'MI': round(np.mean(all_mi), 4), 'MI_std': round(np.std(all_mi, ddof=1), 4),
    'QABF': round(np.mean(all_q), 4), 'QABF_std': round(np.std(all_q, ddof=1), 4),
    'Score': round(np.mean(all_sc), 4), 'Score_std': round(np.std(all_sc, ddof=1), 4),
}

print(f'\n=== V5 Minimal 10-group mean ===')
for k, v in mean.items():
    print(f'{k}: {v}')

# Compare with V5 Full
print(f'\n=== V5 Full reference ===')
print(f'Score: 0.2406±0.026 (1.63M params)')

out = BASE / 'docs' / '数值溯源' / 'verification' / 'v5_minimal_eval.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    json.dump({
        'eval_date': datetime.now().strftime('%Y-%m-%d'),
        'model': 'MSegNetV5Minimal',
        'params': n_params,
        'protocol': 'MultiFocusDataset [0,1] 512x512, 10-group test set',
        'score_formula': 'SF + 0.5 * AG',
        'per_group': per_group,
        'mean_metrics': mean,
        'v5_full_reference': {'Score': 0.2406, 'std': 0.026, 'params': 1630000},
    }, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out}')
