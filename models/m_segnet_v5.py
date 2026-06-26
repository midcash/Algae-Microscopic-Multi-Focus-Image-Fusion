"""
m-SegNet V5 — 原始特征决策 + 置信度感知融合

关键洞察（基于V2-V4实验结果）:
- V2 sharper pseudo: Score+6.5% 但视觉碎片化仍在
- V3/V4 架构改动: 未超越V2
- 根因: DecisionNet 依赖 decoder 特征（经5次下采样+5次上采样，高频纹理丢失）
  → 在边缘密集区无法准确判断"哪个源图最清晰" → 选择错误 → 融合模糊

V5 方案:
1. RawFeatureDecisionNet: 在梯度差异基础上，直接注入原始图像的多尺度Laplacian特征
   → 绕开编解码路径，保留完整高频信息
2. 置信度感知训练: 用 top1-top2 gap 作为置信度，低置信区自动弱化Focal Loss权重
   → 不在不确定区域强行训练，减少错误决策
3. 推理时可选 GapMix（与前代相同）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.depthwise_conv import Stem, Stage
from models.modules.sppf import SPPF
from models.modules.bifpn import BiFPN
from models.modules.simam import SimAM
from models.modules.decision_net import (
    gumbel_softmax_hard, select_and_fuse,
    _gradient_magnitude, bilateral_refine_decision,
)


# ========== RawFeatureDecisionNet ==========
class RawFeatureDecisionNet(nn.Module):
    """
    V5 决策网络: 梯度差异 + 原始图像多尺度Laplacian特征

    核心改进: 不依赖decoder特征，直接用原始图像的多尺度Laplacian
    作为"绝对清晰度"信号注入决策过程
    """

    def __init__(self, num_source_images=5, num_scales=3, base_channels=32):
        super().__init__()
        self.num_source_images = num_source_images
        self.num_scales = num_scales
        num_pairs = num_source_images * (num_source_images - 1) // 2

        # 梯度差异分支（不变）
        grad_in = num_pairs * num_scales  # 30
        self.grad_ops = nn.Sequential(
            nn.Conv2d(grad_in, base_channels, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels, bias=False),
            nn.Conv2d(base_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )

        # V5: 原始图像Laplacian特征（每源2尺度 = 10ch）
        self.register_buffer('lap_k', torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32))
        raw_lap_in = num_source_images * 2  # 5 sources × 2 scales = 10

        self.raw_lap_proj = nn.Sequential(
            nn.Conv2d(raw_lap_in, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2), nn.ReLU(inplace=True),
        )

        # 融合: grad(base_channels) + raw_lap(base_channels/2) → base_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels + base_channels // 2, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(base_channels, num_source_images, 3, padding=1, bias=True)

    def _compute_gradient_diff_feats(self, source_images):
        """多尺度梯度差异特征（与原DecisionNet相同）"""
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
                    diff = torch.abs(grads[i] - grads[j])
                    pair_diffs.append(diff)
            scale_feat = torch.cat(pair_diffs, dim=1)
            if scale < 1.0:
                scale_feat = F.interpolate(scale_feat, size=(H, W), mode='bilinear', align_corners=False)
            all_diff.append(scale_feat)
        return torch.cat(all_diff, dim=1)

    def _compute_raw_lap_feats(self, source_images):
        """原始图像多尺度Laplacian特征"""
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
        return torch.cat(all_lap, dim=1)  # (B, N*2, H, W)

    def forward(self, source_images, decoder_feat=None, decoder_proj=None, decoder_gate=None):
        """
        Args:
            source_images: List[Tensor] (N, B, 3, H, W)
            decoder_feat: 保留接口兼容，V5不使用
        Returns:
            logits: (B, N, H, W)
        """
        # 1. 梯度差异特征
        diff_feats = self._compute_gradient_diff_feats(source_images)
        grad_feat = self.grad_ops(diff_feats)  # (B, base_channels, H, W)

        # 2. 原始Laplacian特征
        raw_lap = self._compute_raw_lap_feats(source_images)
        lap_feat = self.raw_lap_proj(raw_lap)  # (B, base_channels/2, H, W)

        # 3. 融合
        feat = torch.cat([grad_feat, lap_feat], dim=1)
        feat = self.fusion(feat)  # (B, base_channels, H, W)

        # 4. 不再依赖decoder特征

        logits = self.out_conv(feat)
        return logits


# ========== V5 GumbelDecisionFusion ==========
class GumbelDecisionFusionV5(nn.Module):
    """V5 决策融合头"""

    def __init__(self, num_source_images=5, base_channels=32, gumbel_tau=0.67,
                 top_k=1, gap_mix_enabled=False, gap_mix_threshold=0.15, gap_mix_alpha=0.9,
                 mode_refine_enabled=False, mode_refine_threshold=0.15, mode_refine_kernel_size=3,
                 bilateral_refine_enabled=False, bilateral_kernel_size=5,
                 bilateral_sigma_spatial=2.0, bilateral_sigma_color=0.1,
                 use_coarse_prior=False, coarse_prior_strength=0.4,
                 fusion_mode='gumbel'):
        super().__init__()
        self.num_source_images = num_source_images
        self.gumbel_tau = gumbel_tau
        self.top_k = top_k
        self.fusion_mode = fusion_mode
        self.gap_mix_enabled = gap_mix_enabled
        self.gap_mix_threshold = gap_mix_threshold
        self.gap_mix_alpha = gap_mix_alpha
        self.mode_refine_enabled = mode_refine_enabled
        self.mode_refine_threshold = mode_refine_threshold
        self.mode_refine_kernel_size = mode_refine_kernel_size
        self.bilateral_refine_enabled = bilateral_refine_enabled
        self.bilateral_kernel_size = bilateral_kernel_size
        self.bilateral_sigma_spatial = bilateral_sigma_spatial
        self.bilateral_sigma_color = bilateral_sigma_color
        self.use_coarse_prior = use_coarse_prior
        self.coarse_prior_strength = coarse_prior_strength

        # V5: RawFeatureDecisionNet（不需要decoder_proj/decoder_gate）
        self.decision_net = RawFeatureDecisionNet(
            num_source_images=num_source_images,
            num_scales=3, base_channels=base_channels,
        )

    def _top_k_fuse(self, logits, source_images):
        N = len(source_images)
        B, _, H, W = source_images[0].shape
        probs = F.softmax(logits, dim=1)
        grads = [_gradient_magnitude(src) for src in source_images]
        grad_stack = torch.stack(grads, dim=1).squeeze(2)
        weights = probs * (grad_stack + 1e-6).pow(0.3)
        k = min(self.top_k, N)
        thresh = torch.kthvalue(weights, k=N - k + 1, dim=1, keepdim=True)[0]
        mask = (weights >= thresh).float()
        weight_map = weights * mask
        weight_map = weight_map / (weight_map.sum(dim=1, keepdim=True) + 1e-8)
        fused = torch.zeros_like(source_images[0])
        for i in range(N):
            w = weight_map[:, i:i+1].unsqueeze(2).expand(-1, -1, 3, -1, -1)
            fused += w[:, 0] * source_images[i]
        return fused, weight_map

    def _gap_mix_fuse(self, logits, source_images):
        probs = F.softmax(logits, dim=1)
        top2_vals, top2_idx = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
        top1_val = top2_vals[:, 0:1]
        top2_val = top2_vals[:, 1:2] if top2_vals.shape[1] > 1 else torch.zeros_like(top1_val)
        top2_index = top2_idx[:, 1:2] if top2_idx.shape[1] > 1 else top2_idx[:, 0:1]
        top1_index = top2_idx[:, 0:1]
        gap = top1_val - top2_val
        low_conf_mask = (gap < self.gap_mix_threshold).float()
        weight_top1 = (1.0 - low_conf_mask) + low_conf_mask * self.gap_mix_alpha
        weight_top2 = low_conf_mask * (1.0 - self.gap_mix_alpha)
        weight_map = torch.zeros_like(probs)
        weight_map.scatter_(1, top1_index, weight_top1)
        weight_map.scatter_add_(1, top2_index, weight_top2)
        fused = torch.zeros_like(source_images[0])
        for i in range(len(source_images)):
            fused += weight_map[:, i:i+1] * source_images[i]
        return fused, weight_map.unsqueeze(2), gap

    def _mode_refine_decision(self, logits):
        probs = F.softmax(logits, dim=1)
        top2_vals, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
        top1 = top2_vals[:, 0:1]
        top2 = top2_vals[:, 1:2] if top2_vals.shape[1] > 1 else torch.zeros_like(top1)
        gap = top1 - top2
        low_conf = gap < self.mode_refine_threshold
        base_idx = probs.argmax(dim=1, keepdim=True)
        k = max(1, int(self.mode_refine_kernel_size))
        if k % 2 == 0: k += 1
        onehot = F.one_hot(base_idx.squeeze(1), num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        vote_counts = F.avg_pool2d(onehot, kernel_size=k, stride=1, padding=k//2) * (k*k)
        refined_idx = vote_counts.argmax(dim=1, keepdim=True)
        final_idx = torch.where(low_conf, refined_idx, base_idx)
        decision_map = torch.zeros_like(probs).scatter_(1, final_idx, 1.0).unsqueeze(2)
        return decision_map, gap

    def forward(self, decoder_features, source_images, coarse_prior_logits=None):
        # V5: decoder_features ignored — DecisionNet uses raw image features
        logits = self.decision_net(source_images)

        if self.use_coarse_prior and coarse_prior_logits is not None:
            logits = logits + self.coarse_prior_strength * coarse_prior_logits

        if self.top_k > 1:
            if self.training:
                uniform = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-10)
                gumbel = -torch.log(-torch.log(uniform))
                noisy_logits = (logits + gumbel) / self.gumbel_tau
            else:
                noisy_logits = logits
            fused, weight_map = self._top_k_fuse(noisy_logits, source_images)
            decision_map = weight_map.unsqueeze(2)
        else:
            if self.training:
                if self.fusion_mode == 'softmax':
                    probs = F.softmax(logits, dim=1)
                    decision_map = probs.unsqueeze(2)
                    fused = torch.zeros_like(source_images[0])
                    for k in range(self.num_source_images):
                        fused += probs[:, k:k+1] * source_images[k]
                else:
                    decision_map = gumbel_softmax_hard(logits, tau=self.gumbel_tau, dim=1)
                    decision_map = decision_map.unsqueeze(2)
                    fused = select_and_fuse(source_images, decision_map)
            else:
                if self.fusion_mode == 'softmax':
                    probs = F.softmax(logits, dim=1)
                    decision_map = probs.unsqueeze(2)
                    fused = torch.zeros_like(source_images[0])
                    for k in range(self.num_source_images):
                        fused += probs[:, k:k+1] * source_images[k]
                elif self.mode_refine_enabled:
                    decision_map, _ = self._mode_refine_decision(logits)
                    fused = select_and_fuse(source_images, decision_map)
                elif self.gap_mix_enabled:
                    fused, decision_map, _ = self._gap_mix_fuse(logits, source_images)
                elif self.bilateral_refine_enabled:
                    idx = logits.argmax(dim=1, keepdim=True)
                    raw_decision = torch.zeros_like(logits).scatter_(1, idx, 1.0).unsqueeze(2)
                    guide = source_images[0]
                    decision_map = bilateral_refine_decision(
                        raw_decision, guide,
                        kernel_size=self.bilateral_kernel_size,
                        sigma_spatial=self.bilateral_sigma_spatial,
                        sigma_color=self.bilateral_sigma_color,
                    )
                    fused = select_and_fuse(source_images, decision_map)
                else:
                    idx = logits.argmax(dim=1, keepdim=True)
                    decision_map = torch.zeros_like(logits).scatter_(1, idx, 1.0).unsqueeze(2)
                    fused = select_and_fuse(source_images, decision_map)
        return fused, decision_map, logits, decoder_features


# ========== V5 Main Model ==========
class MSegNetV5(nn.Module):
    """m-SegNet V5 — 原始特征决策融合"""

    def __init__(self, num_source_images=5, in_channels=3,
                 stem_channels=24, stage_channels=None, stage_blocks=None,
                 use_bifpn=True, use_simam=True, use_fusion_head='decision',
                 multi_source_bifpn_fusion='mean',
                 bifpn_out_channels=64, bifpn_num_layers=2,
                 decoder_tail_channels=8, cross_source_alpha=0.2,
                 top_k=1, gap_mix_enabled=False, gap_mix_threshold=0.15,
                 gap_mix_alpha=0.9, mode_refine_enabled=False,
                 mode_refine_threshold=0.15, mode_refine_kernel_size=3,
                 bilateral_refine_enabled=False, bilateral_kernel_size=5,
                 bilateral_sigma_spatial=2.0, bilateral_sigma_color=0.1,
                 use_coarse_prior=False, coarse_prior_strength=0.4,
                 coarse_prior_hidden_channels=32,
                 fusion_mode='gumbel',
                 use_sppf=True):
        if stage_channels is None: stage_channels = [24, 48, 96, 128]
        if stage_blocks is None: stage_blocks = [2, 4, 6, 3]
        super().__init__()
        self.num_source_images = num_source_images
        self.multi_source_bifpn_fusion = multi_source_bifpn_fusion
        self.cross_source_alpha = cross_source_alpha
        self.bifpn_out_channels = bifpn_out_channels

        self.encoder = LightEncoderV5(in_channels, stem_channels, stage_channels, stage_blocks)
        self.sppf = SPPF(in_channels=stage_channels[-1], out_channels=stage_channels[-1])
        self.use_bifpn = use_bifpn; self.use_simam = use_simam
        self.use_sppf = use_sppf
        self.bifpn = BiFPN(
            in_channels_list=stage_channels, out_channels=bifpn_out_channels,
            num_levels=len(stage_channels), num_layers=bifpn_num_layers,
        ) if use_bifpn else None
        if use_simam: self.simam = SimAM()
        fusion_input_channels = stage_channels[-1] * num_source_images
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_input_channels, bifpn_out_channels, 1, bias=False),
            nn.BatchNorm2d(bifpn_out_channels), nn.ReLU6(inplace=True),
        )
        decoder_channels = [128, 64, 32, 16, decoder_tail_channels]
        self.coarse_prior = SourceAwareCoarsePriorV5(
            in_channels=stage_channels[-1], hidden_channels=coarse_prior_hidden_channels,
        ) if use_coarse_prior else None
        self.decoder = LightDecoderV5(
            encoder_channels=stage_channels[::-1] + [stem_channels],
            decoder_channels=decoder_channels, bifpn_channels=bifpn_out_channels,
        )
        if use_fusion_head == 'gumbel':
            self.fusion_head = GumbelDecisionFusionV5(
                num_source_images=num_source_images, base_channels=32, gumbel_tau=0.67,
                top_k=top_k, gap_mix_enabled=gap_mix_enabled,
                gap_mix_threshold=gap_mix_threshold, gap_mix_alpha=gap_mix_alpha,
                mode_refine_enabled=mode_refine_enabled,
                mode_refine_threshold=mode_refine_threshold,
                mode_refine_kernel_size=mode_refine_kernel_size,
                bilateral_refine_enabled=bilateral_refine_enabled,
                bilateral_kernel_size=bilateral_kernel_size,
                bilateral_sigma_spatial=bilateral_sigma_spatial,
                bilateral_sigma_color=bilateral_sigma_color,
                use_coarse_prior=use_coarse_prior,
                coarse_prior_strength=coarse_prior_strength,
                fusion_mode=fusion_mode,
            )
        else:
            self.fusion_head = None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _fuse_stage_outputs_for_bifpn(self, encoded_features):
        if self.multi_source_bifpn_fusion == 'first':
            return list(encoded_features[0]['features'][1:])
        stage_outputs = []
        for level in range(len(encoded_features[0]['features']) - 1):
            stacked = torch.stack([enc['features'][level + 1] for enc in encoded_features], dim=0)
            fused_level = stacked.mean(dim=0) if self.multi_source_bifpn_fusion == 'mean' else stacked.max(dim=0)[0]
            stage_outputs.append(fused_level)
        return stage_outputs

    def forward(self, source_images):
        encoded_features = [self.encoder(src) for src in source_images]
        num_features = len(encoded_features[0]['features'])
        for level in range(num_features):
            stacked = torch.stack([f['features'][level] for f in encoded_features], dim=0)
            max_feat, _ = stacked.max(dim=0, keepdim=True)
            for f in encoded_features:
                f['features'][level] = f['features'][level] + self.cross_source_alpha * max_feat.squeeze(0)
        stacked_out = torch.stack([f['out'] for f in encoded_features], dim=0)
        max_out, _ = stacked_out.max(dim=0, keepdim=True)
        for f in encoded_features:
            f['out'] = f['out'] + self.cross_source_alpha * max_out.squeeze(0)
        stage_outputs = self._fuse_stage_outputs_for_bifpn(encoded_features)
        sppf_out = self.sppf(stage_outputs[-1]) if self.use_sppf else stage_outputs[-1]
        if self.use_simam: sppf_out = self.simam(sppf_out)
        stage_outputs[-1] = sppf_out
        if self.use_bifpn and self.bifpn is not None:
            bifpn_features = self.bifpn(stage_outputs)
        else:
            bifpn_features = [torch.zeros(s.shape[0], self.bifpn_out_channels, s.shape[2], s.shape[3],
                                          device=s.device, dtype=s.dtype) for s in stage_outputs]
        concatenated = torch.cat([f['out'] for f in encoded_features], dim=1)
        fused_deep = self.fusion_conv(concatenated)
        avg_encoder_features = [torch.stack([f['features'][i] for f in encoded_features], dim=0).mean(dim=0)
                                for i in range(len(encoded_features[0]['features']))]
        decoded = self.decoder(fused_deep, avg_encoder_features, bifpn_features)
        coarse_prior_logits = None
        if self.coarse_prior is not None:
            deepest_features = [f['out'] for f in encoded_features]
            coarse_prior_logits = self.coarse_prior(deepest_features, target_size=decoded.shape[2:])
        return self.fusion_head(decoded, source_images, coarse_prior_logits=coarse_prior_logits)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_source_images=5, **kwargs):
    return MSegNetV5(num_source_images=num_source_images, **kwargs)


# Re-used encoder/decoder (same as V2)
class LightEncoderV5(nn.Module):
    def __init__(self, in_channels=3, stem_channels=16, stage_channels=None, stage_blocks=None):
        if stage_channels is None: stage_channels = [16, 32, 64, 128]
        if stage_blocks is None: stage_blocks = [2, 3, 4, 3]
        super().__init__()
        self.stem = Stem(in_channels, stem_channels)
        self.stages = nn.ModuleList()
        pc = stem_channels
        for oc, nb in zip(stage_channels, stage_blocks):
            self.stages.append(Stage(pc, oc, nb, stride=2)); pc = oc

    def forward(self, x):
        feats = []; x = self.stem(x); feats.append(x)
        for s in self.stages: x = s(x); feats.append(x)
        return {'out': x, 'features': feats}


class LightDecoderV5(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_skip_connection=True,
                 bifpn_channels=64, num_bifpn_features=4):
        super().__init__()
        self.use_skip_connection = use_skip_connection
        self.bifpn_fusion = nn.ModuleList([
            nn.Sequential(nn.Conv2d(bifpn_channels, dc, 1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True))
            for dc in decoder_channels[:num_bifpn_features]
        ])
        self.blocks = nn.ModuleList(); pc = bifpn_channels
        for i, (ec, dc) in enumerate(zip(encoder_channels, decoder_channels)):
            hb = i < num_bifpn_features
            hs = use_skip_connection and i < len(encoder_channels) - 1
            ic = pc
            if hb:
                ic += dc
            if hs:
                ic += ec
            conv = nn.Sequential(
                nn.Conv2d(ic, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True),
                nn.Conv2d(dc, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True),
            )
            self.blocks.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), conv))
            pc = dc
        self.has_bifpn = [i < num_bifpn_features for i in range(len(decoder_channels))]
        self.has_skip = [use_skip_connection and i < len(encoder_channels) - 1 for i in range(len(decoder_channels))]

    def forward(self, x, encoder_features=None, bifpn_features=None):
        out = x
        for i, block in enumerate(self.blocks):
            inputs = [out]
            if bifpn_features is not None and self.has_bifpn[i] and i < len(bifpn_features):
                bf = F.interpolate(bifpn_features[i], size=out.shape[2:], mode='bilinear', align_corners=False)
                inputs.append(self.bifpn_fusion[i](bf))
            if self.use_skip_connection and encoder_features is not None and self.has_skip[i] and i < len(encoder_features):
                ef = encoder_features[-(i+1)]
                if ef.shape[2:] != out.shape[2:]: ef = F.interpolate(ef, size=out.shape[2:], mode='bilinear', align_corners=False)
                inputs.append(ef)
            out = torch.cat(inputs, dim=1); out = block(out)
        return out


class SourceAwareCoarsePriorV5(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )

    def forward(self, deepest_features, target_size=None):
        coarse_scores = [self.score_head(f) for f in deepest_features]
        coarse_logits = torch.cat(coarse_scores, dim=1)
        if target_size is not None and coarse_logits.shape[2:] != target_size:
            coarse_logits = F.interpolate(coarse_logits, size=target_size, mode='bilinear', align_corners=False)
        return coarse_logits
