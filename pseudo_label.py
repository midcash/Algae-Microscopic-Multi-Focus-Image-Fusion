"""
R13: 像素级伪标签训练

用法:
  verify:   python pseudo_label.py
  pretrain: python pseudo_label.py --mode pretrain --epochs 20 --batch 4
"""
import sys, os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.modules.decision_net import DecisionNet
from utils.data_loader import MultiFocusDataset


# ============================ 伪标签 ============================

@torch.no_grad()
def compute_pseudo_labels(sources, gap_thresh=0.2, temperature=0.3):
    """
    像素级伪标签 — 相对 Laplacian 排名 + 对比度归一化 + gap门控均匀平滑

    对模糊区域（top-2 gap 不够大）使用均匀分布，避免为不可信区域提供强监督。
    Returns: (B, 5, H, W) 软伪标签
    """
    B = sources[0].shape[0]
    H, W = sources[0].shape[2:]
    device = sources[0].device

    lap_k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)
    scores = []

    for scale in [1.0, 0.5]:
        if scale < 1.0:
            scaled = [F.interpolate(s, (int(H*scale), int(W*scale)), mode='bilinear', align_corners=False) for s in sources]
        else:
            scaled = sources
        grads = []
        for s in scaled:
            gray = s.mean(dim=1, keepdim=True)
            lap = F.conv2d(gray, lap_k, padding=1)
            grads.append(lap.abs())
        ss = torch.cat(grads, dim=1)
        if scale < 1.0:
            ss = F.interpolate(ss, (H, W), mode='bilinear', align_corners=False)
        scores.append(ss)

    g = sum(scores) / len(scores)
    g_min = g.min(dim=1, keepdim=True)[0]
    g_max = g.max(dim=1, keepdim=True)[0]
    g_norm = (g - g_min) / (g_max - g_min + 1e-8)

    # 确定区域：sharp softmax；模糊区域：均匀分布
    top2 = torch.topk(g_norm, k=2, dim=1)
    gap = top2[0][:, 0] - top2[0][:, 1]
    uncertain_mask = (gap < gap_thresh).float().unsqueeze(1)  # (B, 1, H, W)

    pseudo_sharp = F.softmax(g_norm / temperature, dim=1)  # (B, 5, H, W)
    pseudo_uniform = torch.full_like(pseudo_sharp, 1.0 / 5)
    pseudo = pseudo_sharp * (1 - uncertain_mask) + pseudo_uniform * uncertain_mask

    return pseudo


# ============================ Focal Loss ============================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        loss = -(targets * ((1 - probs) ** self.gamma) * log_probs).sum()
        return loss / (logits.shape[0] * logits.shape[2] * logits.shape[3])


# ============================ 验证 ============================

def verify():
    print('Verifying pseudo labels...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds = MultiFocusDataset('./all_data/split_data/train', is_train=True)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    for bidx, batch in enumerate(dl):
        sources = [img.to(device) for img in batch['sources']]
        labels = compute_pseudo_labels(sources)
        B, C, H, W = labels.shape
        avg = labels.reshape(B, C, -1).mean(dim=2).mean(dim=0)
        entropy = -(labels * torch.log(labels + 1e-8)).sum(dim=1).mean()
        max_p = labels.max(dim=1)[0]
        print(f'\nBatch {bidx}:')
        for c in range(C):
            print(f'  Class {c+1}: avg={avg[c]:.4f}')
        print(f'  Entropy: {entropy:.4f} (ln5=1.6094)')
        print(f'  Avg max prob: {max_p.mean():.4f}')
        print(f'  Hard >0.5: {(max_p > 0.5).float().mean():.4f}')
        print(f'  Hard >0.8: {(max_p > 0.8).float().mean():.4f}')
        if bidx >= 2:
            break
    print('\nVerify done.')


# ============================ 监督预训练 ============================

def pretrain(args):
    print('=' * 50)
    print(f'  R13: 像素级伪标签监督预训练')
    print(f'  Focal gamma={args.focal_gamma}, pseudo_temp={args.pseudo_temp}')
    print(f'  Epochs={args.epochs}, Batch={args.batch}')
    print('=' * 50)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.output, exist_ok=True)

    train_ds = MultiFocusDataset(args.train_data, is_train=True)
    val_ds = MultiFocusDataset(args.val_data or args.train_data, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
    print(f'Train: {len(train_ds)} groups, Val: {len(val_ds)} groups')

    dnet = DecisionNet(num_source_images=5, num_scales=3, base_channels=32).to(device)
    print(f'DecisionNet params: {sum(p.numel() for p in dnet.parameters())/1e3:.1f}K')

    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(dnet.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = 1e9
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        dnet.train()
        total_loss = 0

        for batch in train_loader:
            sources = [img.to(device) for img in batch['sources']]
            pseudo = compute_pseudo_labels(sources, temperature=args.pseudo_temp)
            logits = dnet(sources)
            loss = criterion(logits, pseudo)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        dnet.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                sources = [img.to(device) for img in batch['sources']]
                pseudo = compute_pseudo_labels(sources, temperature=args.pseudo_temp)
                logits = dnet(sources)
                val_loss += criterion(logits, pseudo).item()
        avg_val_loss = val_loss / len(val_loader)

        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            torch.save(dnet.state_dict(), os.path.join(args.output, 'decision_net_pretrained.pth'))

        elapsed = time.time() - t0
        print(f'Ep {epoch:2d}/{args.epochs}: train={avg_loss:.4f} val={avg_val_loss:.4f} {"[BEST]" if is_best else ""} ({elapsed:.0f}s)')

    print(f'\nDone! Best val_loss={best_loss:.4f}')
    path = os.path.join(args.output, 'decision_net_pretrained.pth')
    print(f'Weights: {path}')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='verify', choices=['verify', 'pretrain'])
    parser.add_argument('--train-data', default='./all_data/split_data/train')
    parser.add_argument('--val-data', default='./all_data/split_data/val')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--pseudo-temp', type=float, default=0.3)
    parser.add_argument('--output', default='./runs/pretrain/supervised_v1')
    args = parser.parse_args()

    if args.mode == 'verify':
        verify()
    else:
        pretrain(args)
