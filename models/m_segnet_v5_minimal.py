"""
m-SegNet V5 Minimal — 精简版（仅保留 DecisionNet + Gumbel 融合头）
去掉编码器、解码器、SPPF、BiFPN、SimAM、fusion_conv
决策路径: 源图 → Sobel差异(30ch) + Laplacian(10ch) → DecisionNet(27K) → Gumbel → 融合图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.decision_net import (
    gumbel_softmax_hard, select_and_fuse,
    _gradient_magnitude,
)


class RawFeatureDecisionNet(nn.Module):
    """V5 决策网络: 梯度差异 + 原始图像多尺度 Laplacian 特征"""

    def __init__(self, num_source_images=5, num_scales=3, base_channels=32):
        super().__init__()
        self.num_source_images = num_source_images
        num_pairs = num_source_images * (num_source_images - 1) // 2

        # 梯度差异分支（30ch）
        grad_in = num_pairs * num_scales
        self.grad_ops = nn.Sequential(
            nn.Conv2d(grad_in, base_channels, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels, bias=False),
            nn.Conv2d(base_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )

        # Raw Laplacian 分支（10ch）
        self.register_buffer('lap_k', torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32))
        raw_lap_in = num_source_images * 2
        self.raw_lap_proj = nn.Sequential(
            nn.Conv2d(raw_lap_in, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2), nn.ReLU(inplace=True),
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels + base_channels // 2, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(base_channels, num_source_images, 3, padding=1, bias=True)

    def _compute_gradient_diff_feats(self, source_images):
        N = len(source_images)
        B, _, H, W = source_images[0].shape
        scales = [1.0, 0.5, 0.25]
        all_diff = []
        for scale in scales:
            if scale < 1.0:
                sH, sW = int(H * scale), int(W * scale)
                scaled = [F.interpolate(src, size=(sH, sW), mode='bilinear', align_corners=False)
                          for src in source_images]
            else:
                scaled = source_images; sH, sW = H, W
            grads = [_gradient_magnitude(src) for src in scaled]
            pair_diffs = []
            for i in range(N):
                for j in range(i + 1, N):
                    pair_diffs.append(torch.abs(grads[i] - grads[j]))
            scale_feat = torch.cat(pair_diffs, dim=1)
            if scale < 1.0:
                scale_feat = F.interpolate(scale_feat, size=(H, W), mode='bilinear', align_corners=False)
            all_diff.append(scale_feat)
        return torch.cat(all_diff, dim=1)

    def _compute_raw_lap_feats(self, source_images):
        N = len(source_images)
        B, _, H, W = source_images[0].shape
        all_lap = []
        for scale in [1.0, 0.5]:
            for src in source_images:
                gray = src.mean(dim=1, keepdim=True)
                if scale < 1.0:
                    sH, sW = int(H * scale), int(W * scale)
                    gray_s = F.interpolate(gray, size=(sH, sW), mode='bilinear', align_corners=False)
                else:
                    gray_s = gray
                lap = F.conv2d(gray_s, self.lap_k, padding=1).abs()
                if scale < 1.0:
                    lap = F.interpolate(lap, size=(H, W), mode='bilinear', align_corners=False)
                all_lap.append(lap)
        return torch.cat(all_lap, dim=1)

    def forward(self, source_images, decoder_feat=None, decoder_proj=None, decoder_gate=None):
        diff_feats = self._compute_gradient_diff_feats(source_images)
        grad_feat = self.grad_ops(diff_feats)
        raw_lap = self._compute_raw_lap_feats(source_images)
        lap_feat = self.raw_lap_proj(raw_lap)
        feat = torch.cat([grad_feat, lap_feat], dim=1)
        feat = self.fusion(feat)
        logits = self.out_conv(feat)
        return logits


class GumbelDecisionFusionV5(nn.Module):
    """V5 决策融合头（精简版，无解码器依赖）"""

    def __init__(self, num_source_images=5, base_channels=32, gumbel_tau=0.67):
        super().__init__()
        self.num_source_images = num_source_images
        self.gumbel_tau = gumbel_tau
        self.decision_net = RawFeatureDecisionNet(
            num_source_images=num_source_images, num_scales=3, base_channels=base_channels)

    def forward(self, decoder_features, source_images, coarse_prior_logits=None):
        logits = self.decision_net(source_images)
        if self.training:
            dm = gumbel_softmax_hard(logits, tau=self.gumbel_tau, dim=1).unsqueeze(2)
            fused = select_and_fuse(source_images, dm)
        else:
            idx = logits.argmax(dim=1, keepdim=True)
            dm = torch.zeros_like(logits).scatter_(1, idx, 1.0).unsqueeze(2)
            fused = select_and_fuse(source_images, dm)
        return fused, dm, logits, decoder_features


class MSegNetV5Minimal(nn.Module):
    """m-SegNet V5 精简版 — 仅 DecisionNet(27K) + Gumbel 融合头"""

    def __init__(self, num_source_images=5, base_channels=32, gumbel_tau=0.67):
        super().__init__()
        self.fusion_head = GumbelDecisionFusionV5(
            num_source_images=num_source_images, base_channels=base_channels, gumbel_tau=gumbel_tau)

    def forward(self, source_images):
        # decoder_features placeholder — DecisionNet ignores this entirely
        B = source_images[0].shape[0]
        dummy_dec = torch.zeros(B, 8, source_images[0].shape[2], source_images[0].shape[3],
                                device=source_images[0].device)
        return self.fusion_head(dummy_dec, source_images)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_source_images=5, **kwargs):
    return MSegNetV5Minimal(num_source_images=num_source_images)
