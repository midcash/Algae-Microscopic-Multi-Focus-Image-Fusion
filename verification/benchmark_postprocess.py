"""Felzenszwalb超像素后处理耗时测试
流程: 全局决策图 → Felzenszwalb分割 → 区域多数投票 → 边缘感知羽化 → 双边滤波
不包含模型推理，仅测后处理
"""
import sys, time, cv2, numpy as np
from pathlib import Path
from skimage.segmentation import felzenszwalb

BASE = Path(__file__).resolve().parent.parent
test_dir = BASE / 'all_data' / 'split_data' / 'test' / 'group_003'

# Load full-res guide image
guide = cv2.imread(str(test_dir / 'img_3.png'))
guide = cv2.cvtColor(guide, cv2.COLOR_BGR2RGB)
h, w = guide.shape[:2]
print(f'Resolution: {w}x{h}')

# Simulate a decision map (random 5-class)
np.random.seed(42)
dec_map = np.random.randint(0, 5, (h, w), dtype=np.int32)

N_WARM = 3; N_TEST = 10
times = {}

# ---- 0. Downsample guide for speed ----
t0 = time.perf_counter()
guide_small = cv2.resize(guide, (w // 4, h // 4))
t1 = time.perf_counter()
print(f'  Guide downsample: {(t1-t0)*1000:.0f} ms')

# ---- 1. Felzenszwalb segmentation ----
for _ in range(N_WARM):
    _ = felzenszwalb(guide_small, scale=200, sigma=0.8, min_size=200)
t0 = time.perf_counter()
for _ in range(N_TEST):
    seg_small = felzenszwalb(guide_small, scale=200, sigma=0.8, min_size=200)
t1 = time.perf_counter()
times['Felzenszwalb'] = (t1 - t0) / N_TEST * 1000

# ---- 2. Upsample segmentation ----
seg = cv2.resize(seg_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
t0 = time.perf_counter()
for _ in range(N_TEST):
    _ = cv2.resize(seg_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
t1 = time.perf_counter()
times['Seg upsample'] = (t1 - t0) / N_TEST * 1000

# ---- 3. Region majority voting ----
n_sp = seg.max() + 1
print(f'  Superpixels: {n_sp}')

# Vectorized version (np.add.at)
t0 = time.perf_counter()
for _ in range(N_TEST):
    votes = np.zeros((n_sp, 5), dtype=np.float64)
    np.add.at(votes, seg.ravel(), np.eye(5, dtype=np.float64)[dec_map.ravel()])
    best = votes.argmax(axis=1)
    dec_seg_vec = best[seg]
t1 = time.perf_counter()
times['Majority voting (vec)'] = (t1 - t0) / N_TEST * 1000

# ---- 4. Edge-aware feathering ----
# Simple implementation: average pooling over boundary mask
feather_width = 20

def edge_feather(dec_seg, guide_img, width=20):
    h, w = dec_seg.shape
    # Dilate superpixel boundaries
    from scipy import ndimage
    # Find boundaries by gradient of decision map
    gy, gx = np.gradient(dec_seg.astype(np.float32))
    edges = (np.abs(gx) + np.abs(gy)) > 0
    # Dilate edges
    kernel = np.ones((width, width), dtype=np.uint8)
    edge_zone = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1)
    return edge_zone

for _ in range(N_WARM):
    _ = edge_feather(dec_seg_vec, guide, feather_width)
t0 = time.perf_counter()
for _ in range(N_TEST):
    edge_zone = edge_feather(dec_seg_vec, guide, feather_width)
t1 = time.perf_counter()
times['Edge feathering'] = (t1 - t0) / N_TEST * 1000

# ---- 5. Bilateral filter ----
# Apply to fused image (simulate with guide)
fused_sim = guide.astype(np.float32)
t0 = time.perf_counter()
for _ in range(N_TEST):
    # Bilateral on each channel
    result = np.zeros_like(fused_sim)
    for c in range(3):
        result[:,:,c] = cv2.bilateralFilter(fused_sim[:,:,c], 5, 30, 30)
t1 = time.perf_counter()
times['Bilateral filter'] = (t1 - t0) / N_TEST * 1000

# ---- 6. Full fusion (select pixels by decision) ----
t0 = time.perf_counter()
for _ in range(N_TEST):
    fused = np.zeros_like(guide, dtype=np.float32)
    for i in range(5):
        fused[dec_seg_vec == i] = guide[dec_seg_vec == i]
t1 = time.perf_counter()
times['Pixel selection'] = (t1 - t0) / N_TEST * 1000

# ===== Summary =====
print(f'\n{"="*50}')
print(f'Post-processing time (5440x3648, {n_sp} SPs)')
print(f'{"="*50}')
for name, t in times.items():
    print(f'  {name:<25s} {t:>8.0f} ms')
total = sum(times.values())
print(f'  {"─"*35}')
print(f'  {"TOTAL post-process":<25s} {total:>8.0f} ms ({total/1000:.1f}s)')
print()
print(f'  Tiled inference:  4408 ms (4.4s)')
print(f'  Post-processing:  {total:.0f} ms ({total/1000:.1f}s)')
print(f'  {"─"*35}')
print(f'  GRAND TOTAL:      {4408+total:.0f} ms ({(4408+total)/1000:.1f}s)')
