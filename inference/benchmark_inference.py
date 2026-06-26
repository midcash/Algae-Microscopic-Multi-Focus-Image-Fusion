import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.m_segnet_v2 import create_model
from utils.data_loader import MultiFocusDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark V2 inference latency')
    parser.add_argument('--model', type=str, required=True, help='Path to checkpoint .pt')
    parser.add_argument('--config', type=str, default=None, help='Optional config.json path; defaults to sibling config.json')
    parser.add_argument('--data', type=str, required=True, help='Dataset directory, e.g. all_data/split_data/test')
    parser.add_argument('--device', type=str, default='cuda', help='cuda / cpu / cuda:0')
    parser.add_argument('--input-size', type=int, default=512, help='Resize input to square size')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for benchmark')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of timed batches')
    parser.add_argument('--warmup-batches', type=int, default=3, help='Warmup batches before timing')
    parser.add_argument('--fp16', action='store_true', help='Enable autocast fp16 on CUDA')
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_arg)
        return torch.device('cpu')
    return torch.device(device_arg)


def load_model_from_config(model_path: Path, config_path: Path, device: torch.device):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    mcfg = cfg['model']
    model = create_model(
        num_source_images=mcfg.get('num_source_images', 5),
        stem_channels=mcfg.get('stem_channels', 24),
        stage_channels=mcfg.get('stage_channels', [24, 48, 96, 128]),
        stage_blocks=mcfg.get('stage_blocks', [2, 4, 6, 3]),
        use_fusion_head=mcfg.get('fusion_head', 'gumbel'),
        multi_source_bifpn_fusion=mcfg.get('multi_source_bifpn_fusion', 'mean'),
        bifpn_out_channels=mcfg.get('bifpn_out_channels', 48),
        bifpn_num_layers=mcfg.get('bifpn_num_layers', 2),
        decoder_tail_channels=mcfg.get('decoder_tail_channels', 12),
        top_k=mcfg.get('top_k', 1),
        cross_source_alpha=mcfg.get('cross_source_alpha', 0.2),
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, mcfg


def main():
    args = parse_args()
    device = resolve_device(args.device)

    model_path = Path(args.model)
    config_path = Path(args.config) if args.config else model_path.parent.parent / 'config.json'
    model, mcfg = load_model_from_config(model_path, config_path, device)

    dataset = MultiFocusDataset(args.data, is_train=False, input_size=args.input_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    warmup = args.warmup_batches
    timed = args.num_batches
    total_ms = 0.0
    total_groups = 0
    total_images = 0
    measured_batches = 0

    autocast_enabled = args.fp16 and device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= warmup + timed:
                break

            source_images = batch['sources']
            source_images = [img.to(device, non_blocking=True) for img in source_images]

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type='cuda', enabled=autocast_enabled):
                _ = model(source_images)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if batch_idx >= warmup:
                total_ms += elapsed_ms
                total_groups += source_images[0].shape[0]
                total_images += source_images[0].shape[0] * len(source_images)
                measured_batches += 1

    effective_batches = max(1, measured_batches)
    avg_batch_ms = total_ms / effective_batches
    avg_group_ms = total_ms / max(1, total_groups)
    avg_source_image_ms = total_ms / max(1, total_images)
    fps_groups = 1000.0 / avg_group_ms

    print('\n' + '=' * 60)
    print('Inference Benchmark Result')
    print('=' * 60)
    print(f'model: {model_path}')
    print(f'config: {config_path}')
    print(f'device: {device}')
    print(f'input_size: {args.input_size}x{args.input_size}')
    print(f'batch_size: {args.batch}')
    print(f'warmup_batches: {warmup}')
    print(f'timed_batches: {effective_batches}')
    print(f'fp16: {autocast_enabled}')
    print(f'num_source_images: {mcfg.get("num_source_images", 5)}')
    print(f'parameter_count: {mcfg.get("parameter_count", "unknown")}')
    print(f'avg_batch_time_ms: {avg_batch_ms:.3f}')
    print(f'avg_group_time_ms: {avg_group_ms:.3f}')
    print(f'avg_source_image_time_ms: {avg_source_image_ms:.3f}')
    print(f'groups_per_second: {fps_groups:.3f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
