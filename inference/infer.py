"""
推理入口脚本

用法:
    python infer.py image1.jpg image2.jpg image3.jpg -o fused.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import cv2
import numpy as np

# 导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.m_segnet_v2 import create_model
from utils.image_enhance import apply_clahe_rgb


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='m-SegNet 多聚焦图像融合推理')

    # 输入
    parser.add_argument('images', type=str, nargs='+', help='输入图像路径')

    # 模型
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='模型权重路径 (默认使用 runs/train/<latest>/checkpoints/best.pt)')
    parser.add_argument('--config', type=str, default=None,
                        help='可选：显式指定 config.json；默认自动从 checkpoint 邻近目录读取')

    # 输出
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出图像路径')

    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='启用 FP16 推理')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='禁用 FP16')

    # 结构参数覆盖（默认优先从 config.json 自动恢复；仅在需要时手动覆盖）
    parser.add_argument('--bifpn-out-channels', type=int, default=None,
                        help='手动覆盖 BiFPN 输出通道数')
    parser.add_argument('--bifpn-num-layers', type=int, default=None,
                        help='手动覆盖 BiFPN 层数')
    parser.add_argument('--decoder-tail-channels', type=int, default=None,
                        help='手动覆盖 decoder tail 通道数')
    parser.add_argument('--multi-source-bifpn-fusion', type=str, default=None,
                        choices=['first', 'mean', 'max'],
                        help='手动覆盖多源 BiFPN 融合方式')
    parser.add_argument('--cross-source-alpha', type=float, default=None,
                        help='手动覆盖 cross_source_alpha')
    parser.add_argument('--top-k', type=int, default=None,
                        help='手动覆盖 top-k 融合设置')

    # 分块处理
    parser.add_argument('--tile', type=int, default=0,
                        help='分块大小 (0=不分块)')

    # 轻量推理端 Top-2 Gap 弱混合
    parser.add_argument('--gap-mix-enabled', action='store_true',
                        help='启用基于 Top-2 Gap 的弱混合推理，仅在 V2 gumbel 头推理时生效')
    parser.add_argument('--gap-mix-threshold', type=float, default=0.15,
                        help='Top-2 gap 小于该阈值时启用弱混合')
    parser.add_argument('--gap-mix-alpha', type=float, default=0.9,
                        help='弱混合时 top-1 源图权重，top-2 权重为 1-alpha')

    # R20-v1: 低置信区域局部多数投票平滑
    parser.add_argument('--mode-refine-enabled', action='store_true',
                        help='启用R20-v1低置信区域决策平滑，仅在V2 gumbel头推理时生效')
    parser.add_argument('--mode-refine-threshold', type=float, default=0.15,
                        help='top-2 gap 低于该阈值时触发局部多数投票平滑')
    parser.add_argument('--mode-refine-kernel-size', type=int, default=3,
                        help='局部多数投票窗口大小，建议3或5')

    # TensorRT
    parser.add_argument('--trt', action='store_true',
                        help='启用 TensorRT 推理')
    parser.add_argument('--trt-engine', type=str, default=None,
                        help='TensorRT Engine 路径')

    # 其他
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='详细输出')
    parser.add_argument('--use-clahe', action='store_true',
                        help='对输入源图启用 CLAHE 亮度增强（藻类弱边界补充预处理）')
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0,
                        help='CLAHE clip limit')
    parser.add_argument('--clahe-tile-size', type=int, default=8,
                        help='CLAHE tile size')

    return parser.parse_args()


def load_image(path, size=512, use_clahe=False, clahe_clip_limit=2.0, clahe_tile_size=8):
    """加载并预处理图像"""
    # OpenCV 读取 (BGR)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Failed to load image: {path}')

    # 转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if use_clahe:
        img = apply_clahe_rgb(img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_size)

    # 调整大小
    if size > 0:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # 归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0

    # HWC -> CHW -> Tensor
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    return img


def save_image(tensor, path, original_path=None):
    """保存图像"""
    # Tensor -> numpy
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # RGB -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 获取原始图像尺寸
    if original_path:
        orig = cv2.imread(original_path)
        if orig is not None:
            img = cv2.resize(img, (orig.shape[1], orig.shape[0]))

    cv2.imwrite(path, img)
    print(f'Saved: {path}')


@torch.no_grad()
def fuse_images(model, image_paths, device, fp16=False, tile_size=0, verbose=False,
                use_clahe=False, clahe_clip_limit=2.0, clahe_tile_size=8):
    """融合多张图像"""
    # 加载图像
    if verbose:
        print(f'Loading {len(image_paths)} images...')

    images = []
    use_fp16 = fp16 and device.type == 'cuda'
    input_dtype = torch.float16 if use_fp16 else torch.float32
    for path in image_paths:
        img = load_image(
            path,
            size=512 if tile_size == 0 else 0,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size,
        )
        images.append(img.to(device=device, dtype=input_dtype))

    if verbose:
        print(f'Input shapes: {[img.shape for img in images]}')

    # 分块处理
    if tile_size > 0:
        return fuse_tiled(model, images, tile_size, device, fp16, verbose)

    # 推理
    model.eval()

    with torch.autocast(device_type='cuda', enabled=fp16 and device.type == 'cuda'):
        start = time.time()
        fused = model(images)[0]  # model returns (fused, decision_map, logits, decoder_feat)
        elapsed = time.time() - start

    if verbose:
        print(f'Inference time: {elapsed*1000:.1f}ms')

    return fused


@torch.no_grad()
def fuse_tiled(model, images, tile_size, device, fp16, verbose):
    """分块融合（用于大图）"""
    b, c, h, w = images[0].shape

    # 创建输出张量
    fused = torch.zeros_like(images[0])

    # 分块
    tiles_y = (h + tile_size - 1) // tile_size
    tiles_x = (w + tile_size - 1) // tile_size

    if verbose:
        print(f'Processing {tiles_y}x{tiles_x} tiles...')

    model.eval()

    start = time.time()

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * tile_size
            y1 = min((ty + 1) * tile_size, h)
            x0 = tx * tile_size
            x1 = min((tx + 1) * tile_size, w)

            # 提取 tile
            tile_images = [img[:, :, y0:y1, x0:x1] for img in images]

            # 调整到 512x512
            tile_images = [
                F.interpolate(img, size=(512, 512), mode='bilinear')
                for img in tile_images
            ]

            # 推理
            with torch.autocast(device_type='cuda', enabled=fp16 and device.type == 'cuda'):
                tile_fused = model(tile_images)[0]  # model returns 4-tuple

            # 调整回原尺寸
            tile_fused = F.interpolate(tile_fused, size=(y1-y0, x1-x0), mode='bilinear')

            # 写入输出
            fused[:, :, y0:y1, x0:x1] = tile_fused

    elapsed = time.time() - start

    if verbose:
        print(f'Tiled inference time: {elapsed*1000:.1f}ms')

    return fused


def find_best_model():
    """自动查找最佳模型"""
    import glob

    # 查找最近的训练目录
    train_dirs = glob.glob('./runs/train/*')
    if not train_dirs:
        return None

    # 按时间排序
    train_dirs.sort(key=os.path.getmtime, reverse=True)

    # 查找最佳检查点
    for train_dir in train_dirs:
        best_path = os.path.join(train_dir, 'checkpoints', 'best.pt')
        if os.path.exists(best_path):
            return best_path

    return None


def resolve_config_path(model_path: Path, config_arg: str | None) -> Path | None:
    if config_arg:
        return Path(config_arg)
    candidate = model_path.parent.parent / 'config.json'
    return candidate if candidate.exists() else None


def build_model_kwargs_from_config(config_path: Path | None, num_source_images: int, args):
    model_kwargs = {
        'num_source_images': num_source_images,
        'use_fusion_head': 'gumbel',
        'gap_mix_enabled': args.gap_mix_enabled,
        'gap_mix_threshold': args.gap_mix_threshold,
        'gap_mix_alpha': args.gap_mix_alpha,
        'mode_refine_enabled': args.mode_refine_enabled,
        'mode_refine_threshold': args.mode_refine_threshold,
        'mode_refine_kernel_size': args.mode_refine_kernel_size,
    }

    if config_path and config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        mcfg = cfg.get('model', {})
        model_kwargs.update({
            'stem_channels': mcfg.get('stem_channels', 24),
            'stage_channels': mcfg.get('stage_channels', [24, 48, 96, 128]),
            'stage_blocks': mcfg.get('stage_blocks', [2, 4, 6, 3]),
            'multi_source_bifpn_fusion': mcfg.get('multi_source_bifpn_fusion', 'mean'),
            'bifpn_out_channels': mcfg.get('bifpn_out_channels', 64),
            'bifpn_num_layers': mcfg.get('bifpn_num_layers', 2),
            'decoder_tail_channels': mcfg.get('decoder_tail_channels', 8),
            'top_k': mcfg.get('top_k', 1),
            'cross_source_alpha': mcfg.get('cross_source_alpha', 0.2),
        })

    # CLI 手动覆盖
    if args.multi_source_bifpn_fusion is not None:
        model_kwargs['multi_source_bifpn_fusion'] = args.multi_source_bifpn_fusion
    if args.bifpn_out_channels is not None:
        model_kwargs['bifpn_out_channels'] = args.bifpn_out_channels
    if args.bifpn_num_layers is not None:
        model_kwargs['bifpn_num_layers'] = args.bifpn_num_layers
    if args.decoder_tail_channels is not None:
        model_kwargs['decoder_tail_channels'] = args.decoder_tail_channels
    if args.cross_source_alpha is not None:
        model_kwargs['cross_source_alpha'] = args.cross_source_alpha
    if args.top_k is not None:
        model_kwargs['top_k'] = args.top_k

    return model_kwargs


def main():
    args = parse_args()

    # 检查输入
    if len(args.images) < 2:
        print('Error: At least 2 input images required')
        sys.exit(1)

    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'
    device = torch.device(args.device)

    # 输出路径
    if args.output is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output = f'./output/fused_{timestamp}.png'
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # 加载模型
    model_path = args.model or find_best_model()
    if model_path is None:
        print('Error: No model found. Please specify with --model')
        sys.exit(1)

    print(f'Loading model from {model_path}...')
    model_path = Path(model_path)
    config_path = resolve_config_path(model_path, args.config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 创建模型（优先从 config.json 恢复结构，必要时允许 CLI 覆盖）
    model_kwargs = build_model_kwargs_from_config(config_path, len(args.images), args)
    model = create_model(**model_kwargs)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f'Model loaded (epoch {checkpoint.get("epoch", "?")})')
    if config_path is not None and config_path.exists():
        print(f'Loaded architecture config from {config_path}')
    else:
        print('No config.json found; using default V2 structure plus CLI overrides')
    if args.gap_mix_enabled:
        print(f'Gap-mix enabled: threshold={args.gap_mix_threshold}, alpha={args.gap_mix_alpha}')
    if args.mode_refine_enabled:
        print(f'R20-v1 mode-refine enabled: threshold={args.mode_refine_threshold}, kernel={args.mode_refine_kernel_size}')

    # FP16
    if args.fp16 and device.type == 'cuda':
        print('Using FP16 inference')
        # 半精度
        model.half()

    # TensorRT (TODO: 实现)
    if args.trt:
        print('TensorRT support: coming soon')

    # 融合
    print(f'Fusing {len(args.images)} images...')
    fused = fuse_images(
        model, args.images, device,
        fp16=args.fp16,
        tile_size=args.tile,
        verbose=args.verbose,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=args.clahe_tile_size,
    )

    # 保存
    # 使用第一张输入图像的尺寸作为参考
    save_image(fused, args.output, args.images[0])

    # 打印统计
    if args.verbose:
        print(f'\nOutput statistics:')
        print(f'  Min: {fused.min().item():.4f}')
        print(f'  Max: {fused.max().item():.4f}')
        print(f'  Mean: {fused.mean().item():.4f}')


if __name__ == '__main__':
    main()
