"""SD-Fuse 适配5源藻类数据 — 顺序两两融合"""
import os, sys, argparse, numpy as np
from pathlib import Path
from PIL import Image
import torch

# Add both SD-Fuse dir and project root to path
SD_DIR = Path(__file__).parent
ROOT = SD_DIR.parent.parent
sys.path.insert(0, str(SD_DIR))
sys.path.insert(0, str(ROOT))

from test import load_model, infer_dm, fuse
from utils.metrics import spatial_frequency, average_gradient, mutual_information, qabf

def fuse_5_pairwise(model, paths_5, device, pad_mult=16):
    """顺序两两融合: src1+src2→tmp, tmp+src3→tmp... 以512x512运行"""
    import torch.nn.functional as F
    imgs = []
    for p in paths_5:
        img = Image.open(p).convert("RGB")
        img = img.resize((512, 512), Image.BILINEAR)
        t = np.asarray(img, dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(t).permute(2,0,1).unsqueeze(0).to(device))

    current = imgs[0].clone().detach()
    for i in range(1, len(imgs)):
        dm = infer_dm(model, current, imgs[i], pad_mult=pad_mult)
        current = fuse(current, imgs[i], dm).detach()
        del dm; torch.cuda.empty_cache()
    return current

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True, help='test data dir, e.g. all_data/split_data/test')
    ap.add_argument('--output', default='output/sd_fuse_results')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device, strict=True)

    import cv2
    data_dir = Path(args.data)
    groups = sorted([d.name for d in data_dir.iterdir() if d.name.startswith('group_')])

    all_sf, all_ag, all_mi, all_q = [], [], [], []
    for g in groups:
        paths = [str(data_dir / g / f'img_{i}.png') for i in range(1, 6)]
        fused_t = fuse_5_pairwise(model, paths, device)
        fused_np = fused_t[0].cpu().clamp(0,1).permute(1,2,0).numpy()
        fused_uint8 = (fused_np * 255).astype(np.uint8)

        out_path = Path(args.output) / f'{g}_fused.png'
        Image.fromarray(fused_uint8).save(str(out_path))

        # Evaluate
        f_gray = fused_np.mean(axis=2)
        srcs_gray = []
        for p in paths:
            s = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            s_g = cv2.resize(s.mean(axis=2), (f_gray.shape[1], f_gray.shape[0]))
            srcs_gray.append(s_g.astype(np.float32)/255.0)

        sf = spatial_frequency(f_gray)
        ag = average_gradient(f_gray)
        mi = mutual_information(f_gray, srcs_gray)
        qs = [qabf(f_gray, srcs_gray[a], srcs_gray[b]) for a in range(5) for b in range(a+1,5)]
        q = np.mean(qs)
        score = sf + 0.5 * ag
        print(f'{g}: SF={sf:.4f} AG={ag:.4f} QABF={q:.4f} Score={score:.4f}')
        all_sf.append(sf); all_ag.append(ag); all_mi.append(mi); all_q.append(q)

    print(f'\n10-group avg: SF={np.mean(all_sf):.4f} AG={np.mean(all_ag):.4f} MI={np.mean(all_mi):.4f} QABF={np.mean(all_q):.4f} Score={np.mean(all_sf)+0.5*np.mean(all_ag):.4f}')

if __name__ == '__main__':
    main()
