"""
训练入口脚本（Step 2 — 架构清理版）

改动（相对 Step 1 / R18）：
1. 清理遗留参数：--val-split, --use-algae-decision-prior, --decision-prior-*
   --warmup-epochs, --config, --tensorboard, --scheduler
2. 统一 loss 接口：模型返回 4 元组 (fused, decision_map, logits, decoder_feat)
3. 移除 pretrain-decision 独立模式（已由 pretrain_only.py 独立管理）
4. decoder_proj 投影不再作为可选——直接集成在 GumbelDecisionFusion 中
5. Step 2 新增: decoder_features 参与 loss 决策（--decoder-consistency-weight）

用法:
    python train.py --data ./all_data/train --epochs 50 --batch 4
    python train.py --data ./all_data/train --val_data ./all_data/split_data/val --epochs 50 --batch 4
    python train.py --data ./all_data/train --epochs 50 --batch 4 --pseudo-label-weight 1.0 --focal-gamma 2.0
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.losses.combined_loss import CombinedLoss
from utils.data_loader import MultiFocusDataset
from utils.metrics import spatial_frequency, average_gradient, mutual_information, qabf


def _import_model(version='v2'):
    """动态导入指定版本的模型"""
    if version == 'v2':
        from models.m_segnet_v2 import create_model, count_parameters
    elif version == 'v3':
        from models.m_segnet_v3 import create_model, count_parameters
    elif version == 'v4':
        from models.m_segnet_v4 import create_model, count_parameters
    elif version == 'v5':
        from models.m_segnet_v5 import create_model, count_parameters
    elif version == 'v6':
        from models.m_segnet_v6 import create_model, count_parameters
    else:
        raise ValueError(f"Unknown model version: {version}")
    return create_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='训练 m-SegNet V2 — Gumbel 决策融合')
    # 数据
    parser.add_argument('--data', type=str, required=True, help='训练数据目录')
    parser.add_argument('--val_data', type=str, default=None, help='验证数据目录（不传则使用 --data 划分）')
    parser.add_argument('--input-size', type=int, default=512, help='训练时源图缩放尺寸（默认512）')

    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--early-stop', type=int, default=15)

    # 输出
    parser.add_argument('--output', type=str, default='./runs/train')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--model-version', type=str, default='v2', choices=['v2', 'v3', 'v4', 'v5', 'v6'],
                        help='模型版本: v2/v3/v4/v5/v6')

    # 恢复/预训练
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--load-pretrained-decision', type=str, default=None)

    # AMP
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', dest='amp', action='store_false')

    # Loss 组件权重（R16_pure 默认 = ssim=0, tv=0, pseudo=1.0, contrast=1.0）
    parser.add_argument('--ssim-weight', type=float, default=0.0)
    parser.add_argument('--grad-match-weight', type=float, default=0.0)
    parser.add_argument('--tv-weight', type=float, default=0.0)
    parser.add_argument('--contrast-weight', type=float, default=1.0)

    # Focal / 伪标签
    parser.add_argument('--pseudo-label-weight', type=float, default=1.0)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--pseudo-gap-thresh', type=float, default=0.2)
    parser.add_argument('--pseudo-temperature', type=float, default=0.3)
    parser.add_argument('--pseudo-window-size', type=int, default=0,
                        help='V7: 伪标签区域一致性窗口(0=不启用, 建议8-16)')
    parser.add_argument('--oracle-weight', type=float, default=0.0,
                        help='Oracle Guided Loss权重(替代/补充伪标签)')
    parser.add_argument('--oracle-window-size', type=int, default=8,
                        help='Oracle区域窗口大小')

    # 扩展 loss（R17a/R17b）
    parser.add_argument('--edge-mag-weight', type=float, default=0.0)
    parser.add_argument('--orientation-weight', type=float, default=0.0)

    # 融合决策
    parser.add_argument('--top-k', type=int, default=1,
                        help='推理时 top-k 加权融合（R18）')
    parser.add_argument('--fusion-mode', type=str, default='gumbel', choices=['gumbel', 'softmax'],
                        help='融合模式: gumbel=Gumbel-STE硬决策, softmax=Softmax加权混合(消融)')
    parser.add_argument('--bifpn-out-channels', type=int, default=64,
                        help='BiFPN 输出通道数（V2 架构消融）')
    parser.add_argument('--bifpn-num-layers', type=int, default=2,
                        help='BiFPN 堆叠层数（V2 架构消融）')
    parser.add_argument('--multi-source-bifpn-fusion', type=str, default='first',
                        choices=['first', 'mean', 'max'],
                        help='多源特征供 BiFPN 前的融合方式（V2 架构消融）')
    parser.add_argument('--decoder-tail-channels', type=int, default=8,
                        help='解码器尾部通道数（V2 架构消融）')
    parser.add_argument('--cross-source-alpha', type=float, default=0.1,
                        help='跨源增强残差系数 alpha（V2 架构消融）')
    parser.add_argument('--use-sppf', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='启用SPPF模块 (default: true, 消融用 --use-sppf false)')
    parser.add_argument('--use-bifpn', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='启用BiFPN模块 (default: true, 消融用 --use-bifpn false)')
    parser.add_argument('--use-simam', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='启用SimAM模块 (default: true, 消融用 --use-simam false)')
    parser.add_argument('--use-coarse-prior', action='store_true',
                        help='R21-min：启用source-aware coarse decision prior')
    parser.add_argument('--coarse-prior-strength', type=float, default=0.4,
                        help='R21-min：coarse prior logits 对最终决策logits的偏置强度')
    parser.add_argument('--coarse-prior-hidden-channels', type=int, default=32,
                        help='R21-min：coarse prior轻量打分头的隐藏通道数')

    # Step 2: decoder_features 参与 loss 监督
    parser.add_argument('--decoder-consistency-weight', type=float, default=0.0,
                        help='decoder 特征与 Laplacian 清晰度排序的一致性 loss 权重')
    parser.add_argument('--low-conf-consistency-weight', type=float, default=0.0,
                        help='R20-v3-lite：低置信区域局部一致性 loss 权重')
    parser.add_argument('--low-conf-gap-thresh', type=float, default=0.15,
                        help='R20-v3-lite：top1-top2 gap 小于该阈值时施加一致性约束')
    parser.add_argument('--low-conf-kernel-size', type=int, default=3,
                        help='R20-v3-lite：局部一致性窗口大小，建议 3 或 5')

    # R22: Edge-Aware TV & Bilateral Smooth (auto-experiment)
    parser.add_argument('--edge-aware-tv-weight', type=float, default=0.0,
                        help='R22：Edge-Aware TV loss 权重，边缘区放款TV，平坦区惩罚碎片')
    parser.add_argument('--edge-aware-tv-threshold', type=float, default=0.1,
                        help='R22：边缘判定阈值（归一化梯度值）')
    parser.add_argument('--edge-aware-tv-decay', type=float, default=3.0,
                        help='R22：边缘区TV衰减指数')
    parser.add_argument('--bilateral-smooth-weight', type=float, default=0.0,
                        help='R22：Bilateral平滑loss权重，颜色相近的相邻像素决策应一致')
    parser.add_argument('--bilateral-sigma-range', type=float, default=0.1,
                        help='R22：双边平滑的颜色sigma参数')

    return parser.parse_args()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def make_logger(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message='', flush=False):
        print(message, flush=flush)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
            if flush:
                f.flush()

    return log


def compute_metrics(fused, sources, device):  # 融合后的图像张量，多张源图像张量、计算设备
    """计算无参考融合指标 — [0,1] 浮点灰度输入"""
    fused_np = fused.detach().cpu().numpy() # 张量->numpy数组: detach 切断计算图 ， 把数据移到CPU，转为numpy数组
    sources_np = [s.detach().cpu().numpy() for s in sources] # 多张源图像张量->numpy数组

    total_sf, total_ag, total_qabf, total_mi = 0, 0, 0, 0 # 累加批次内的所有图像的指标值
    batch_size = fused.shape[0]  # 一次计算多少图
    num_sources = len(sources)   # 源图像数量（如2张：红外+可见光）

    for b in range(batch_size):
        # 取出当前批次的单张融合图 + 单张源图
        img = fused_np[b]
        srcs = [s[b] for s in sources_np]

        # 灰度化：彩色图转灰度图（RGB取均值），灰度图保持不变
        img_gray = np.mean(img, axis=0) if img.ndim == 3 else img
        srcs_gray = [np.mean(s, axis=0) if s.ndim == 3 else s for s in srcs]

        # 1. 空间频率 SF：衡量图像清晰度、细节丰富度
        total_sf += spatial_frequency(img_gray)

        # 2. 平均梯度 AG：衡量图像纹理、边缘清晰程度（越大越清晰）
        total_ag += average_gradient(img_gray)

        qabf_sum = 0.0
        cnt = 0

        # 遍历所有源图像对（避免重复计算：如源1+源2，只算1次）
        for a in range(num_sources):
            for b2 in range(a + 1, num_sources):
                qabf_sum += qabf(img_gray, srcs_gray[a], srcs_gray[b2])
                cnt += 1
        total_qabf += qabf_sum / max(cnt, 1) # 求平均，防止除0

        # 4. 互信息 MI：衡量融合图与源图的信息相关性（越大保留信息越多）
        total_mi += mutual_information(img_gray, srcs_gray)

    return {
        'sf': total_sf / batch_size,
        'ag': total_ag / batch_size,
        'qabf': total_qabf / batch_size,
        'mi': total_mi / batch_size,
        'combined_score': total_sf / batch_size + 0.5 * total_ag / batch_size,
    }

# 模型进入评估模式
# 关闭梯度计算
# 遍历验证集图片
# 模型生成融合图
# 计算损失
# 计算你之前写的 SF/AG/QABF/MI 融合指标
# 所有批次求平均
# 返回损失 + 指标
def validate(model, val_loader, criterion, device): # 你的图像融合模型；验证集数据加载器（批量加载图片）；损失函数（计算融合图的监督损失）；运行设备 cuda / cpu
    """验证 — 模型返回 4 元组 (fused 最终融合图像（最重要）, decision_map 决策图（模型选择哪个源图的权重）, logits 模型输出的原始预测, decoder_features 解码器特征（用于损失计算）
)"""
    model.eval() # 把模型设为评估模式
    total_loss = 0
    all_metrics = [] # total_loss：累加整个验证集的总损失； all_metrics：保存每张图的融合指标（SF/AG/QABF/MI）

    with torch.no_grad(): # 关闭梯度计算；加速推理、节省显存；验证阶段必须加（不训练就不需要计算梯度）
        for batch in val_loader:
            source_images = [img.to(device) for img in batch['sources']]# 从 batch 中取出多张源图像（比如 2 张 / 5 张）；把所有图片搬到 device（GPU/CPU）上
            fused, decision_map, logits, decoder_feat = model(source_images)  # 模型前向推理 输入：多张源图 输出：4 个结果（融合图、决策图、logits、解码器特征）
            loss = criterion(fused, source_images, decision_map, logits,
                             decoder_features=decoder_feat) #调用损失函数计算损失 损失会衡量融合图是否符合预期
            total_loss += loss.item()# 把当前批次的损失累加到总损失 .item()：把张量转成普通数字
            metrics = compute_metrics(fused, source_images, device) # 调用了之前的指标计算函数
            all_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]} # 对所有批次的指标求平均值 得到整个验证集的平均融合质量
    return total_loss / len(val_loader), avg_metrics # 平均验证损失；平均融合指标（用于判断融合效果）


# 参数解析 → 实验环境创建 → 数据集加载 → 模型构建 → 训练循环 → 验证 → 保存模型 → 早停。
def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    exp_name = args.name or f'step2_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exp_dir = Path(args.output) / exp_name # 创建实验文件夹，按时间命名，保存日志和模型
    exp_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(exp_dir / 'train.log')
    metrics_history_path = exp_dir / 'metrics_history.jsonl'
    if metrics_history_path.exists() and not args.resume:
        metrics_history_path.unlink()
    log(f'Device: {device}')
    log(f'Experiment: {exp_dir}')

    # ========================================================================
    # 数据
    # ========================================================================
    train_dataset = MultiFocusDataset(args.data, is_train=True, input_size=args.input_size)
    if args.val_data:
        val_dataset = MultiFocusDataset(args.val_data, is_train=False, input_size=args.input_size)
    else:
        val_dataset = MultiFocusDataset(args.data, is_train=False, input_size=args.input_size)

    nw = min(args.workers, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
    )

    # ========================================================================
    # 模型
    # ========================================================================
    create_model_fn, count_params_fn = _import_model(args.model_version)
    log(f'Creating model (version={args.model_version})...')
    model_kwargs = dict(
        num_source_images=5,
        use_fusion_head='gumbel',
        top_k=args.top_k,
        bifpn_out_channels=args.bifpn_out_channels,
        bifpn_num_layers=args.bifpn_num_layers,
        multi_source_bifpn_fusion=args.multi_source_bifpn_fusion,
        decoder_tail_channels=args.decoder_tail_channels,
        cross_source_alpha=args.cross_source_alpha,
        use_coarse_prior=args.use_coarse_prior,
        coarse_prior_strength=args.coarse_prior_strength,
        coarse_prior_hidden_channels=args.coarse_prior_hidden_channels,
        use_sppf=args.use_sppf,
        use_bifpn=args.use_bifpn,
        use_simam=args.use_simam,
    )
    if args.model_version in ('v2', 'v5'):
        model_kwargs['fusion_mode'] = args.fusion_mode
    model = create_model_fn(**model_kwargs).to(device)

    if args.load_pretrained_decision:
        load_path = args.load_pretrained_decision
        log(f'Loading pretrained DecisionNet from: {load_path}')
        state_dict = torch.load(load_path, map_location=device, weights_only=True)
        if hasattr(model, 'fusion_head') and hasattr(model.fusion_head, 'decision_net'): # hasattr() 是 Python 内置函数，专门用来判断一个对象是否拥有指定的属性 / 方法，返回 True 或 False。
            model.fusion_head.decision_net.load_state_dict(state_dict)
            log(f'  Loaded DecisionNet ({len(state_dict)} keys)')
        else:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            log(f'  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    total_params = count_params_fn(model)  # 统计模型总参数量
    log(f'Total params: {total_params / 1e6:.2f}M')

    model_config = {
        'class': model.__class__.__name__,
        'num_source_images': 5,
        'stem_channels': 24,
        'stage_channels': [24, 48, 96, 128],
        'stage_blocks': [2, 4, 6, 3],
        'bifpn_out_channels': args.bifpn_out_channels,
        'bifpn_num_layers': args.bifpn_num_layers,
        'decoder_tail_channels': args.decoder_tail_channels,
        'multi_source_bifpn_fusion': args.multi_source_bifpn_fusion,
        'fusion_head': 'gumbel',
        'gumbel_tau': 0.67,
        'top_k': args.top_k,
        'cross_source_alpha': args.cross_source_alpha,
        'use_coarse_prior': args.use_coarse_prior,
        'coarse_prior_strength': args.coarse_prior_strength,
        'coarse_prior_hidden_channels': args.coarse_prior_hidden_channels,
        'parameter_count': total_params,
    }
    write_json(exp_dir / 'config.json', {
        'args': vars(args),
        'model': model_config,
    })

    # ========================================================================
    # Loss + Optimizer
    # ========================================================================
    # 复合损失函数（SSIM、梯度、边缘、对比度、伪标签等）
    criterion = CombinedLoss(
        ssim_weight=args.ssim_weight, grad_match_weight=args.grad_match_weight,
        grad_contrast_weight=args.contrast_weight, tv_weight=args.tv_weight,
        pseudo_label_weight=args.pseudo_label_weight, focal_gamma=args.focal_gamma,
        gap_thresh=args.pseudo_gap_thresh, pseudo_temperature=args.pseudo_temperature,
        pseudo_window_size=args.pseudo_window_size,
        oracle_weight=args.oracle_weight, oracle_window_size=args.oracle_window_size,
        edge_mag_weight=args.edge_mag_weight,
        orientation_weight=args.orientation_weight,
        decoder_consistency_weight=args.decoder_consistency_weight,
        low_conf_consistency_weight=args.low_conf_consistency_weight,
        low_conf_gap_thresh=args.low_conf_gap_thresh,
        low_conf_kernel_size=args.low_conf_kernel_size,
        edge_aware_tv_weight=args.edge_aware_tv_weight,
        edge_aware_tv_threshold=args.edge_aware_tv_threshold,
        edge_aware_tv_decay=args.edge_aware_tv_decay,
        bilateral_smooth_weight=args.bilateral_smooth_weight,
        bilateral_sigma_range=args.bilateral_sigma_range,
    )

    # AdamW：优化器
    # CosineAnnealingLR：余弦退火学习率（逐渐降低）
    # GradScaler：自动混合精度训练（加速）
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda') if args.amp else None

    # ========================================================================
    # 恢复
    # ========================================================================
    start_epoch = 1
    best_score = 0.0
    early_stop_counter = 0

# 如果训练中断，可以从上次保存的 epoch 继续训练。
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_score = ckpt.get('best_score', 0.0)
        log(f'Resumed from epoch {ckpt["epoch"]}')

    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 训练循环
    # ========================================================================
    log(f'\nTraining {args.epochs} epochs... (early stop: {args.early_stop})')
    # 把 5 张源图送入模型
    # 模型输出：融合图 + 决策图 + 特征
    # 计算复合损失
    # 反向传播更新模型权重
    # 梯度裁剪防止梯度爆炸

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            source_images = [img.to(device) for img in batch['sources']]

            def training_step():
                """统一的前向+loss 计算，返回 (loss_value, loss_obj) 用于 backward"""
                fused, decision_map, logits, decoder_feat = model(source_images)
                loss = criterion(fused, source_images, decision_map, logits,
                                 decoder_features=decoder_feat)
                return loss

            if scaler:
                with autocast('cuda'):
                    loss = training_step()
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.accum_steps == 0:
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss = training_step()
                loss.backward()
                if (batch_idx + 1) % args.accum_steps == 0:
                    if args.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item()

            if (batch_idx + 1) % max(1, args.log_interval) == 0:
                log(f'Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}', flush=True)

        avg_train_loss = epoch_loss / len(train_loader)

        # 验证
        # 每训练完一个 epoch，自动验证
        # 计算：SF / AG / MI / QABF
        # 得到 combined_score 综合指标
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        combined_score = val_metrics['combined_score']
        is_best = combined_score > best_score

        if is_best:
            best_score = combined_score
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_score,
            }, ckpt_dir / 'best.pt')
        else:
            early_stop_counter += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0

        summary_line = (
            f'Epoch {epoch}/{args.epochs}: '
            f'Train Loss: {avg_train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'SF: {val_metrics["sf"]:.4f} | '
            f'AG: {val_metrics["ag"]:.4f} | '
            f'MI: {val_metrics["mi"]:.4f} | '
            f'QABF: {val_metrics["qabf"]:.4f} | '
            f'Score: {combined_score:.4f} | '
            f'LR: {current_lr:.8f} | '
            f'Time: {epoch_time:.1f}s '
            f'{"[BEST]" if is_best else ""}'
        )
        log(summary_line)

        metrics_record = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'sf': val_metrics['sf'],
            'ag': val_metrics['ag'],
            'mi': val_metrics['mi'],
            'qabf': val_metrics['qabf'],
            'score': combined_score,
            'lr': current_lr,
            'epoch_time_sec': epoch_time,
            'is_best': is_best,
        }

        if hasattr(criterion, '_last_details') and criterion._last_details:
            d = criterion._last_details
            parts = [f'Focal={d["pseudo_focal"]:.4f}']
            if d['ssim'] > 0: parts.append(f'SSIM={d["ssim"]:.4f}')
            if d['grad_match'] > 0: parts.append(f'Grad={d["grad_match"]:.4f}')
            if d['contrast'] > 0: parts.append(f'Contrast={d["contrast"]:.4f}')
            if d['tv'] > 0: parts.append(f'TV={d["tv"]:.4f}')
            if d['edge_mag'] > 0: parts.append(f'EdgeMag={d["edge_mag"]:.4f}')
            if d['decoder_cons'] > 0: parts.append(f'DecCons={d["decoder_cons"]:.4f}')
            if d['low_conf_cons'] > 0: parts.append(f'LowConfCons={d["low_conf_cons"]:.4f}')
            if d.get('edge_aware_tv', 0) > 0: parts.append(f'EdgeAwareTV={d["edge_aware_tv"]:.4f}')
            if d.get('bilateral_smooth', 0) > 0: parts.append(f'BilateralSmooth={d["bilateral_smooth"]:.4f}')
            if d.get('oracle_loss', 0) > 0: parts.append(f'Oracle={d["oracle_loss"]:.4f}')
            loss_details_line = f'  Loss details: {" | ".join(parts)}'
            log(loss_details_line)
            metrics_record['loss_details'] = d

        append_jsonl(metrics_history_path, metrics_record)

        # epoch 检查点
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'best_score': best_score,
        }, ckpt_dir / f'epoch_{epoch}.pt')

        if args.early_stop > 0 and early_stop_counter >= args.early_stop:
            log(f'Early stopped at epoch {epoch} (no improvement for {args.early_stop} epochs)')
            break

    log(f'\nTraining complete! Best combined_score: {best_score:.4f}')
    log(f'Model saved to: {exp_dir}')


if __name__ == '__main__':
    main()
