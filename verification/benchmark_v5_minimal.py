"""V5 Minimal 端到端推理速度测试
覆盖: 512×512 单次 / 全分辨率瓦片 / CPU / GPU
对比: V5 Full (5/28日志: 66ms模型/850ms端到端)
"""
import sys, time, torch, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from models.m_segnet_v5_minimal import create_model, count_parameters

BASE = Path(__file__).resolve().parent.parent
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEV}')

# Load model with trained weights
ckpt = torch.load(
    str(BASE / 'runs/train/auto_05_v5_raw_feature/checkpoints/best.pt'),
    map_location='cpu', weights_only=True)
dn_weights = {k: v for k, v in ckpt['model'].items() if 'decision_net' in k}
model = create_model()
model.load_state_dict(dn_weights, strict=False)
print(f'Params: {count_parameters(model):,} (0.027M)')

# ===== Test 1: 512x512 GPU =====
model.to(DEV).eval()
test_dir = BASE / 'all_data' / 'split_data' / 'test' / 'group_003'
srcs_np = []
for i in range(1, 6):
    img = cv2.imread(str(test_dir / f'img_{i}.png'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    srcs_np.append(img)

N_WARM = 10; N_TEST = 50
times_load, times_fwd, times_convert, times_total = [], [], [], []

for run in range(N_WARM + N_TEST):
    # 1. Load + preprocess
    t0 = time.perf_counter()
    tensors = []
    for img in srcs_np:
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tensors.append(t.to(DEV))
    t1 = time.perf_counter()

    # 2. Model forward
    with torch.no_grad():
        fused, dm, _, _ = model(tensors)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    # 3. Convert to numpy
    result = fused[0].permute(1, 2, 0).cpu().numpy()
    result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    t3 = time.perf_counter()

    if run >= N_WARM:
        times_load.append((t1 - t0) * 1000)
        times_fwd.append((t2 - t1) * 1000)
        times_convert.append((t3 - t2) * 1000)
        times_total.append((t3 - t0) * 1000)

print(f'\n===== 512x512 GPU =====')
print(f'  Load + preprocess:  {np.mean(times_load):.1f} ms')
print(f'  Model forward:      {np.mean(times_fwd):.1f} ms')
print(f'  Convert to numpy:   {np.mean(times_convert):.1f} ms')
print(f'  TOTAL end-to-end:   {np.mean(times_total):.1f} ms')
print(f'  FPS:                {1000/np.mean(times_total):.1f}')

# ===== Test 2: 512x512 CPU =====
model.cpu()
srcs_tensors_cpu = []
for img in srcs_np:
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    srcs_tensors_cpu.append(t)

with torch.no_grad():
    for _ in range(5): _ = model(srcs_tensors_cpu)
    t0 = time.perf_counter()
    for _ in range(30):
        fused = model(srcs_tensors_cpu)[0]
        _ = fused[0].permute(1, 2, 0).numpy()
    t1 = time.perf_counter()

cpu_total = (t1 - t0) / 30 * 1000
print(f'\n===== 512x512 CPU =====')
print(f'  TOTAL end-to-end:   {cpu_total:.1f} ms')
print(f'  FPS:                {1000/cpu_total:.1f}')

# ===== Test 3: Full resolution tiled (5440x3648) GPU =====
print(f'\n===== 5440x3648 GPU (tiled 512x512) =====')
full_srcs = []
for i in range(1, 6):
    img = cv2.imread(str(test_dir / f'img_{i}.png'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    full_srcs.append(img)
h, w = full_srcs[0].shape[:2]
print(f'  Resolution: {w}x{h}')

model.to(DEV).eval()

t0 = time.perf_counter()
tile_size = 512
ty = (h + tile_size - 1) // tile_size
tx = (w + tile_size - 1) // tile_size
global_dec = np.zeros((h, w), dtype=np.int32)

for yi in range(ty):
    for xi in range(tx):
        y0 = yi * tile_size; y1 = min(y0 + tile_size, h)
        x0 = xi * tile_size; x1 = min(x0 + tile_size, w)
        tiles = []
        for src in full_srcs:
            t = cv2.resize(src[y0:y1, x0:x1], (512, 512))
            tiles.append(torch.from_numpy(t.astype(np.float32)/255.0)
                        .permute(2,0,1).unsqueeze(0).to(DEV))
        with torch.no_grad():
            dm = model(tiles)[1]
            dec = dm.squeeze(2).argmax(dim=1)[0].cpu().numpy()
            dec_full = cv2.resize(dec.astype(np.float32), (x1-x0, y1-y0),
                                  interpolation=cv2.INTER_NEAREST)
            global_dec[y0:y1, x0:x1] = dec_full.astype(np.int32)

# Fuse
fused_full = np.zeros_like(full_srcs[0], dtype=np.float32)
for i in range(5):
    fused_full[global_dec == i] = full_srcs[i][global_dec == i]
t1 = time.perf_counter()
tiled_time = (t1 - t0) * 1000

print(f'  Tiles: {ty}x{tx} = {ty*tx}')
print(f'  Inference + fuse:  {tiled_time:.1f} ms')
print(f'  ({tiled_time/1000:.1f}s)')

# ===== Summary =====
print(f'\n{"="*60}')
print(f'SUMMARY')
print(f'{"="*60}')
print(f'  Model:               V5 Minimal (27K params)')
print(f'  512 GPU end-to-end:  {np.mean(times_total):.1f} ms  ({1000/np.mean(times_total):.0f} FPS)')
print(f'  512 CPU end-to-end:  {cpu_total:.1f} ms  ({1000/cpu_total:.0f} FPS)')
print(f'  5440 tiled infer:    {tiled_time:.1f} ms  ({tiled_time/1000:.1f}s)')
print()
print(f'  V5 Full reference:   66ms model / 850ms e2e / 16.7s full-res')
print(f'  V5 Minimal speedup:  {66/np.mean(times_fwd):.1f}x (model forward)')
