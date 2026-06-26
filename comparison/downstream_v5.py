"""V5 下游验证 — 替代 downstream_validation.py (原来用R16/V2)"""
import sys, torch, numpy as np, cv2, json
from pathlib import Path
sys.path.insert(0, '..')
from models.m_segnet_v5 import create_model
from utils.data_loader import MultiFocusDataset
from torch.utils.data import DataLoader

DEV = torch.device('cuda')
CKPT = 'runs/train/auto_05_v5_raw_feature/checkpoints/best.pt'
TEST = 'all_data/split_data/test'
OUT = Path('runs/downstream/downstream_v5_results.json')
N_SRC = 5

# Load V5
cfg = json.load(open('runs/train/auto_05_v5_raw_feature/config.json'))
m = cfg.get('model', {})
model = create_model(
    num_source_images=5, use_fusion_head='gumbel', top_k=1,
    bifpn_out_channels=m.get('bifpn_out_channels', 64),
    bifpn_num_layers=m.get('bifpn_num_layers', 2),
    decoder_tail_channels=m.get('decoder_tail_channels', 8),
    multi_source_bifpn_fusion=m.get('multi_source_bifpn_fusion', 'mean'),
    cross_source_alpha=m.get('cross_source_alpha', 0.1),
)
ckpt = torch.load(CKPT, map_location=DEV, weights_only=True)
model.load_state_dict(ckpt['model'], strict=False)
model.to(DEV).eval()

dataset = MultiFocusDataset(TEST, is_train=False)
print(f'V5 loaded. Test groups: {len(dataset)}')

# ==== Task 1: SIFT ====
sift = cv2.SIFT_create()
v5_sift, best_sift, avg_sift = [], [], []

for idx, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
    sources = [s.cpu()[0].numpy().transpose(1,2,0) for s in batch['sources']]
    sources_u8 = [(np.clip(s, 0, 1)*255).astype(np.uint8) for s in sources]

    # Average
    avg_img = np.mean(sources_u8, axis=0).astype(np.uint8)
    avg_sift.append(len(sift.detect(cv2.cvtColor(avg_img, cv2.COLOR_RGB2GRAY))))

    # Best single (max SIFT among 5 sources)
    best_s = max(len(sift.detect(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY))) for s in sources_u8)
    best_sift.append(best_s)

    # V5
    srcs_t = [torch.from_numpy(s).permute(2,0,1).unsqueeze(0).float().to(DEV) for s in sources]
    with torch.no_grad():
        fused = model(srcs_t)[0].cpu()[0].numpy().transpose(1,2,0)
    fused_u8 = (np.clip(fused, 0, 1)*255).astype(np.uint8)
    v5_sift.append(len(sift.detect(cv2.cvtColor(fused_u8, cv2.COLOR_RGB2GRAY))))

mu_v5, mu_best, mu_avg = np.mean(v5_sift), np.mean(best_sift), np.mean(avg_sift)
print(f'\n=== Task 1: SIFT Keypoints ===')
print(f'  V5:         {mu_v5:.1f}')
print(f'  Best single: {mu_best:.1f}')
print(f'  Average:     {mu_avg:.1f}')
print(f'  V5 vs Best:  {(mu_v5-mu_best)/mu_best*100:+.1f}%')
print(f'  V5 vs Avg:   {(mu_v5-mu_avg)/mu_avg*100:+.1f}%')

# ==== Task 2: Focus coverage (Laplacian-based) ====
from scipy.signal import convolve2d
v5_cov, best_cov = [], []

for idx, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
    sources = [s.cpu()[0].numpy().transpose(1,2,0) for s in batch['sources']]
    sources_u8 = [(np.clip(s, 0, 1)*255).astype(np.uint8) for s in sources]

    # Focus measure per source
    def focus_map(img_u8):
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(float)
        return np.abs(cv2.Laplacian(gray, cv2.CV_64F))

    best_fm = max(np.mean(focus_map(s)) for s in sources_u8)
    best_cov.append(best_fm)

    # V5
    srcs_t = [torch.from_numpy(s).permute(2,0,1).unsqueeze(0).float().to(DEV) for s in sources]
    with torch.no_grad():
        fused = model(srcs_t)[0].cpu()[0].numpy().transpose(1,2,0)
    fused_u8 = (np.clip(fused, 0, 1)*255).astype(np.uint8)
    v5_cov.append(np.mean(focus_map(fused_u8)))

mu_v5c, mu_bestc = np.mean(v5_cov), np.mean(best_cov)
print(f'\n=== Task 2: Focus Coverage ===')
print(f'  V5:         {mu_v5c:.1f}')
print(f'  Best single: {mu_bestc:.1f}')
print(f'  V5 vs Best:  {(mu_v5c-mu_bestc)/mu_bestc*100:+.1f}%')

# ==== Task 3: Edge density (Canny) ====
v5_ed, best_ed, avg_ed = [], [], []

for idx, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
    sources = [s.cpu()[0].numpy().transpose(1,2,0) for s in batch['sources']]
    sources_u8 = [(np.clip(s, 0, 1)*255).astype(np.uint8) for s in sources]

    def edge_density(img_u8):
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges.sum() / edges.size * 100

    avg_img = np.mean(sources_u8, axis=0).astype(np.uint8)
    avg_ed.append(edge_density(avg_img))
    best_ed.append(max(edge_density(s) for s in sources_u8))

    srcs_t = [torch.from_numpy(s).permute(2,0,1).unsqueeze(0).float().to(DEV) for s in sources]
    with torch.no_grad():
        fused = model(srcs_t)[0].cpu()[0].numpy().transpose(1,2,0)
    fused_u8 = (np.clip(fused, 0, 1)*255).astype(np.uint8)
    v5_ed.append(edge_density(fused_u8))

mu_v5e, mu_beste, mu_avge = np.mean(v5_ed), np.mean(best_ed), np.mean(avg_ed)
print(f'\n=== Task 3: Edge Density ===')
print(f'  V5:         {mu_v5e:.2f}%')
print(f'  Best single: {mu_beste:.2f}%')
print(f'  Average:     {mu_avge:.2f}%')
print(f'  V5 vs Best:  {(mu_v5e-mu_beste)/mu_beste*100:+.1f}%')
print(f'  V5 vs Avg:   {(mu_v5e-mu_avge)/mu_avge*100:+.1f}%')

# Save
result = {
    'model': 'V5 (auto_05_v5_raw_feature)',
    'sift': {'v5_mean': float(mu_v5), 'best_single_mean': float(mu_best),
             'avg_mean': float(mu_avg), 'delta_vs_best_pct': round((mu_v5-mu_best)/mu_best*100, 1)},
    'focus_coverage': {'v5_mean': float(mu_v5c), 'best_single_mean': float(mu_bestc),
                       'delta_vs_best_pct': round((mu_v5c-mu_bestc)/mu_bestc*100, 1)},
    'edge_density': {'v5_pct': round(float(mu_v5e), 2), 'best_single_pct': round(float(mu_beste), 2),
                     'avg_pct': round(float(mu_avge), 2), 'delta_vs_best_pct': round((mu_v5e-mu_beste)/mu_beste*100, 1)},
}
json.dump(result, open(OUT, 'w'), indent=2, ensure_ascii=False)
print(f'\nSaved: {OUT}')
