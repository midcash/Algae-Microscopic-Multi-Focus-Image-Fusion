"""
模型导出脚本

用法:
    python export.py -m ./checkpoints/best.pt -o model.onnx --format onnx
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.m_segnet import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='导出 m-SegNet 模型')

    parser.add_argument('-m', '--model', type=str, required=True, help='PyTorch 模型路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('-f', '--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'trt'], help='导出格式')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset 版本')
    parser.add_argument('--fp16', action='store_true', default=True, help='FP16 量化')
    parser.add_argument('--batch', type=int, default=1, help='批次大小')
    parser.add_argument('--input-size', type=int, default=512, help='输入尺寸')
    parser.add_argument('--num-sources', type=int, default=5, help='源图像数量')

    return parser.parse_args()


def to_torchscript(model, dummy_input, output_path, fp16=False):
    """导出为 TorchScript"""
    print('Exporting to TorchScript...')

    if fp16:
        model.half()
        dummy_input = [d.half() for d in dummy_input] if isinstance(dummy_input, list) else dummy_input.half()

    traced = torch.jit.trace(model, dummy_input)
    traced.save(output_path)

    print(f'Saved: {output_path}')


def to_onnx(model, dummy_input, output_path, opset=11, fp16=False):
    """导出为 ONNX"""
    print('Exporting to ONNX...')

    dynamic_axes = {
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
    for i in range(len(dummy_input)):
        dynamic_axes[f'input_{i}'] = {0: 'batch_size', 2: 'height', 3: 'width'}

    input_names = [f'input_{i}' for i in range(len(dummy_input))]

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=['output'],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    print(f'Saved: {output_path}')


def to_tensorrt(model, dummy_input, output_path, fp16=True, workspace=2):
    """导出为 TensorRT (需要 pycuda 和 tensorrt)"""
    print('Exporting to TensorRT...')

    try:
        import tensorrt as trt
    except ImportError:
        print('Error: tensorrt not installed. Please install: pip install tensorrt')
        sys.exit(1)

    # 先导出 ONNX 作为中间格式
    onnx_path = output_path.replace('.engine', '.onnx')
    to_onnx(model, dummy_input, onnx_path, fp16=fp16)

    # 使用 trtexec 命令行工具转换
    import subprocess

    fp16_flag = '--fp16' if fp16 else ''
    cmd = f'trtexec --onnx={onnx_path} --saveEngine={output_path} --workspace={workspace*1024} {fp16_flag}'

    print(f'Running: {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f'Error: {result.stderr}')
        sys.exit(1)

    print(f'Saved: {output_path}')

    # 清理中间文件
    os.remove(onnx_path)


def main():
    args = parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载模型
    print(f'Loading model from {args.model}...')
    checkpoint = torch.load(args.model, map_location=device)

    model = create_model(
        num_source_images=args.num_sources,
        use_bifpn=True,
        use_simam=True
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f'Model loaded (epoch {checkpoint.get("epoch", "?")})')

    # 创建虚拟输入
    batch_size = args.batch
    input_size = args.input_size

    dummy_input = [
        torch.randn(batch_size, 3, input_size, input_size, device=device)
        for _ in range(args.num_sources)
    ]

    # 测试推理
    print('Testing inference...')
    with torch.no_grad():
        output = model(dummy_input)
    print(f'Output shape: {output.shape}')

    # 导出
    start = time.time()

    if args.format == 'onnx':
        to_onnx(model, dummy_input, args.output, opset=args.opset, fp16=args.fp16)
    elif args.format == 'torchscript':
        to_torchscript(model, dummy_input, args.output, fp16=args.fp16)
    elif args.format == 'trt':
        to_tensorrt(model, dummy_input, args.output, fp16=args.fp16)

    elapsed = time.time() - start
    print(f'Export completed in {elapsed:.1f}s')

    # 验证导出
    print('\nVerifying export...')

    if args.format == 'onnx':
        try:
            import onnx
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            print('ONNX model is valid')
        except ImportError:
            print('onnx not installed, skipping validation')

    # 性能对比
    print('\nPerformance comparison:')

    # PyTorch
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100 * 1000
    print(f'PyTorch: {torch_time:.2f}ms')


if __name__ == '__main__':
    main()
