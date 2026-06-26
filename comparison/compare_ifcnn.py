"""
轻量 IFCNN 对比脚本
===================

目标：在不占 GPU、不保存大量图片、不额外制造系统盘压力的前提下，
对测试集运行 IFCNN 并输出与论文同口径的融合指标。

用法：
  python compare_ifcnn.py
  python compare_ifcnn.py --groups group_003 group_013
  python compare_ifcnn.py --output output/ifcnn_metrics.json
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

BASE = Path(__file__).resolve().parent.parent
IFCNN_CODE = BASE / 'tools' / 'IFCNN' / 'Code'
TEST_DATA = BASE / 'all_data' / 'split_data' / 'test'
DEFAULT_OUTPUT = BASE / 'output' / 'ifcnn_metrics.json'
N_SRC = 5
DEVICE = torch.device('cpu')

sys.path.insert(0, str(IFCNN_CODE))
sys.path.insert(0, str(BASE))

from model import myIFCNN  # noqa: E402
from utils.metrics import spatial_frequency, average_gradient, mutual_information, qabf  # noqa: E402

_mytransforms_spec = importlib.util.spec_from_file_location(
    'ifcnn_myTransforms', IFCNN_CODE / 'utils' / 'myTransforms.py'
)
_mytransforms = importlib.util.module_from_spec(_mytransforms_spec)
assert _mytransforms_spec.loader is not None
_mytransforms_spec.loader.exec_module(_mytransforms)
denorm = _mytransforms.denorm


def parse_args():
    parser = argparse.ArgumentParser(description='Run lightweight IFCNN comparison on test set')
    parser.add_argument('--groups', nargs='+', default=None, help='Optional subset of group_xxx names')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT), help='JSON output path')
    parser.add_argument('--input-size', type=int, default=512, help='Resize source images to square size before IFCNN inference')
    return parser.parse_args()


def find_weight_file() -> Path:
    snapshots = IFCNN_CODE / 'snapshots'
    preferred = snapshots / 'IFCNN-MAX.pth'
    if preferred.exists():
        return preferred
    candidates = sorted(snapshots.glob('*.pth'))
    if not candidates:
        raise FileNotFoundError(f'No IFCNN weight file found under {snapshots}')
    return candidates[0]


def load_ifcnn():
    weight_path = find_weight_file()
    model = myIFCNN(fuse_scheme=0)
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, weight_path


def load_group(group_dir: Path, input_size: int):
    source_tensors = []
    source_uint8 = []
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for k in range(1, N_SRC + 1):
        path = None
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            candidate = group_dir / f'img_{k}{ext}'
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(f'Missing img_{k} in {group_dir}')

        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
        source_uint8.append(img_rgb)

        pil = Image.fromarray(img_rgb)
        tensor = tfm(pil).unsqueeze(0)
        source_tensors.append(tensor)

    return source_tensors, source_uint8


@torch.no_grad()
def sequential_fusion_ifcnn(model, source_tensors):
    current = source_tensors[0].to(DEVICE)
    for i in range(1, len(source_tensors)):
        current = model(current, source_tensors[i].to(DEVICE))

    fused = denorm([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], current[0]).clamp(0, 1)
    fused_np = fused.permute(1, 2, 0).cpu().numpy()
    return fused_np


def compute_metrics(fused_np, source_uint8):
    fused = np.clip(fused_np.astype(np.float32), 0.0, 1.0)
    sources = [np.clip(src.astype(np.float32) / 255.0, 0.0, 1.0) for src in source_uint8]

    fused_gray = np.mean(fused, axis=2) if fused.ndim == 3 else fused
    sources_gray = [np.mean(src, axis=2) if src.ndim == 3 else src for src in sources]

    sf = spatial_frequency(fused_gray)
    ag = average_gradient(fused_gray)
    mi = mutual_information(fused_gray, sources_gray)

    qabf_vals = []
    for i in range(len(sources_gray)):
        for j in range(i + 1, len(sources_gray)):
            qabf_vals.append(qabf(fused_gray, sources_gray[i], sources_gray[j]))
    qabf_mean = float(np.mean(qabf_vals))

    return {
        'SF': float(sf),
        'AG': float(ag),
        'MI': float(mi),
        'QABF': qabf_mean,
        'Score': float(sf + 0.5 * ag),
    }


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, weight_path = load_ifcnn()

    groups = sorted([
        d.name for d in TEST_DATA.iterdir()
        if d.is_dir() and d.name.startswith('group_')
    ])
    if args.groups:
        groups = args.groups

    print('=' * 70)
    print('IFCNN 轻量对比评估')
    print('=' * 70)
    print(f'weights: {weight_path}')
    print(f'device: {DEVICE}')
    print(f'groups: {len(groups)}')
    print(f'input_size: {args.input_size}x{args.input_size}')
    print('no image saving, metrics only')
    print()

    per_group = {}
    all_metrics = []

    for group_name in groups:
        group_dir = TEST_DATA / group_name
        source_tensors, source_uint8 = load_group(group_dir, args.input_size)
        fused_np = sequential_fusion_ifcnn(model, source_tensors)
        metrics = compute_metrics(fused_np, source_uint8)
        per_group[group_name] = metrics
        all_metrics.append(metrics)

        print(
            f'{group_name}: '
            f'SF={metrics["SF"]:.4f}  AG={metrics["AG"]:.4f}  '
            f'MI={metrics["MI"]:.4f}  QABF={metrics["QABF"]:.4f}  '
            f'Score={metrics["Score"]:.4f}'
        )

    mean_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0]
    }

    print('\n' + '=' * 70)
    print('IFCNN 平均结果（测试集）')
    print('=' * 70)
    for key in ['SF', 'AG', 'MI', 'QABF', 'Score']:
        print(f'{key}: {mean_metrics[key]:.4f}')

    payload = {
        'method': 'IFCNN',
        'strategy': 'sequential_pairwise_fusion_for_5_sources',
        'device': str(DEVICE),
        'weights': str(weight_path),
        'groups': groups,
        'mean_metrics': mean_metrics,
        'per_group_metrics': per_group,
        'notes': {
            'input_size': f'{args.input_size}x{args.input_size}',
            'qabf_method': 'pairwise mean over C(5,2)=10 source pairs',
            'score_formula': 'SF + 0.5 * AG',
            'no_images_saved': True,
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'\nSaved metrics JSON to: {output_path}')


if __name__ == '__main__':
    main()
