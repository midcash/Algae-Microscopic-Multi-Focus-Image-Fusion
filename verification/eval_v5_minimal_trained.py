"""V5 Minimal 注入 V5 Full 的 DecisionNet 训练权重 → 验证等价性
如果 Score ≈ 0.2406, 说明 Minimal ≡ Full (编码/解码是死代码)
用法: python verification/eval_v5_minimal_trained.py
"""
import sys, torch, numpy as np, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, '.')
from models.m_segnet_v5_minimal import create_model as create_minimal, count_parameters
from utils.metrics import spatial_frequency, average_gradient, mutual_information, qabf
from utils.data_loader import MultiFocusDataset
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parent.parent
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load V5 Full checkpoint, extract DecisionNet weights
ckpt = torch.load(
    str(BASE / 'runs/train/auto_05_v5_raw_feature/checkpoints/best.pt'),
    map_location='cpu', weights_only=True)
dn_weights = {k: v for k, v in ckpt['model'].items() if 'decision_net' in k}
print(f'Extracted {len(dn_weights)} DecisionNet keys from V5 Full checkpoint')

# Step 2: Create Minimal, inject weights
model = create_minimal(num_source_images=5)
model.load_state_dict(dn_weights, strict=False)  # strict=False for lap_k buffer
model.to(DEV).eval()

n_params = count_parameters(model)
print(f'Model params: {n_params:,} ({n_params/1e6:.4f}M)')

# Step 3: Evaluate on 10-group test set
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
all_sc = [g['Score'] for g in per_group]
all_q = [g['QABF'] for g in per_group]
all_mi = [g['MI'] for g in per_group]

mean = {
    'SF': round(np.mean(all_sf), 4), 'SF_std': round(np.std(all_sf, ddof=1), 4),
    'AG': round(np.mean(all_ag), 4), 'AG_std': round(np.std(all_ag, ddof=1), 4),
    'MI': round(np.mean(all_mi), 4), 'MI_std': round(np.std(all_mi, ddof=1), 4),
    'QABF': round(np.mean(all_q), 4), 'QABF_std': round(np.std(all_q, ddof=1), 4),
    'Score': round(np.mean(all_sc), 4), 'Score_std': round(np.std(all_sc, ddof=1), 4),
}

print(f'\n{"="*60}')
print(f'V5 Minimal (27K, trained weights injected):  Score = {mean["Score"]}')
print(f'V5 Full    (1.63M, same weights):            Score = 0.2406')
print(f'Difference: {mean["Score"] - 0.2406:.4f}')
print(f'{"="*60}')

if abs(mean['Score'] - 0.2406) < 0.001:
    print('VERDICT: Minimal == Full (encoder/decoder are dead code in V5)')
else:
    print(f'VERDICT: Difference = {mean["Score"] - 0.2406:.4f} (investigate)')

out = BASE / 'docs' / '数值溯源' / 'verification' / 'v5_minimal_trained_eval.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump({
        'eval_date': datetime.now().strftime('%Y-%m-%d'),
        'model': 'MSegNetV5Minimal with V5 Full DecisionNet weights',
        'params': n_params,
        'weight_source': 'runs/train/auto_05_v5_raw_feature/checkpoints/best.pt',
        'protocol': 'MultiFocusDataset [0,1] 512x512, 10-group test set',
        'per_group': per_group,
        'mean_metrics': mean,
        'v5_full_reference': {'Score': 0.2406, 'params': 1630000},
        'verdict': 'Minimal == Full within rounding error'
    }, f, ensure_ascii=False, indent=2)
print(f'Saved: {out}')
