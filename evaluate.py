"""
模型评估脚本

用法:
    python evaluate.py --model ./checkpoints/best.pt --data ./all_data/test
"""

import argparse
import os
import sys
import json
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.m_segnet import create_model, count_parameters
from utils.data_loader import create_dataloader, MultiFocusDataset
from utils.metrics import evaluate_fusion


def parse_args():
    parser = argparse.ArgumentParser(description='评估 m-SegNet 模型')

    parser.add_argument('-m', '--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data', type=str, required=True, help='测试数据目录')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('-o', '--output', type=str, default=None, help='结果输出路径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化报告')
    parser.add_argument('--use-algae-decision-prior', action='store_true',
                        help='按 Stage 3 结构创建模型')
    parser.add_argument('--decision-prior-mode', type=str, default='gradient',
                        choices=['gradient', 'gradient_contrast'],
                        help='决策先验图模式')
    parser.add_argument('--decision-prior-fusion', type=str, default='concat',
                        choices=['concat', 'mix'],
                        help='决策先验图接入方式')
    parser.add_argument('--decision-prior-strength', type=float, default=0.5,
                        help='决策先验图混合强度，仅在 mix 模式下生效')

    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg.isdigit():
        gpu_id = int(device_arg)
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cpu')

    if device_arg.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_arg)
        return torch.device('cpu')

    return torch.device(device_arg)


def main():
    args = parse_args()

    # 设备
    device = resolve_device(args.device)
    print(f'Using device: {device}')

    # 加载模型
    print(f'Loading model from {args.model}...')
    checkpoint = torch.load(args.model, map_location=device)

    model = create_model(
        num_source_images=5,
        use_algae_decision_prior=args.use_algae_decision_prior,
        decision_prior_mode=args.decision_prior_mode,
        decision_prior_fusion=args.decision_prior_fusion,
        decision_prior_strength=args.decision_prior_strength
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    num_params = count_parameters(model)
    print(f'Model parameters: {num_params / 1e6:.2f}M')

    # 数据加载
    print(f'Loading test data from {args.data}...')
    test_dataset = MultiFocusDataset(args.data, is_train=False)
    test_loader = create_dataloader(
        test_dataset, batch_size=args.batch, shuffle=False
    )

    # 评估
    print('Evaluating...')

    with torch.no_grad():
        metrics = evaluate_fusion(model, test_loader, device, num_samples=50)

    # 打印结果
    print('\n' + '='*50)
    print('评估结果')
    print('='*50)

    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')

    print('='*50)

    # 保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        result = {
            'model': args.model,
            'dataset': args.data,
            'timestamp': datetime.now().isoformat(),
            'params_m': num_params / 1e6,
            'metrics': metrics
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Results saved to {args.output}')

    # 可视化
    if args.visualize:
        print('Generating visualization...')
        # TODO: 实现可视化


if __name__ == '__main__':
    main()
