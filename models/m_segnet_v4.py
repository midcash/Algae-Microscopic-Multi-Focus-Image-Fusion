"""
m-SegNet V4 — 浅层跳跃决策架构 (Shallow-Skip Decision)

V2 + 一项关键改进:
1. ShallowSkipToDecision: 将编码器 Stage1 特征（128×128, 24ch, 高分辨率纹理）
   直接跳跃连接到 DecisionNet，让决策网络能利用浅层高分辨率纹理信息
   来判断"哪个源图在这里最清晰"

原理: V2 的 DecisionNet 只接收 decoder 尾部特征（8ch, 经过5次下采样+5次上采样），
      高频纹理信息严重丢失 → 无法精确判断局部清晰度 → 选择错误源图 → 融合模糊

V4 修复: 绕过编解码路径，直接将浅层高分辨率特征注入 DecisionNet
        → 保留纹理细节 → 更准确的源图选择 → 融合结果更清晰

目标: 提升 SF/AG，让融合结果接近源图清晰度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.depthwise_conv import Stem, Stage
from models.modules.sppf import SPPF
from models.modules.bifpn import BiFPN
from models.modules.simam import SimAM
from models.modules.decision_net import (
    DecisionNet, gumbel_softmax_hard, select_and_fuse,
    _gradient_magnitude, bilateral_refine_decision,
)


# ========== ShallowSkipToDecision ==========
class ShallowSkipToDecision(nn.Module):
    """
    将编码器浅层特征（高分辨率、富纹理）投影并注入 DecisionNet
    解决深层特征纹理丢失导致的决策不准问题
    """

    def __init__(self, shallow_channels_list=[24, 48], out_channels=32):
        """
        Args:
            shallow_channels_list: 要跳跃连接的编码器 stage 通道数
            out_channels: 投影后的通道数
        """
        super().__init__()
        total_in = sum(shallow_channels_list)
        self.project = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.shallow_channels_list = shallow_channels_list

    def forward(self, shallow_features):
        """
        Args:
            shallow_features: List[List[Tensor]] 各源图的浅层特征
                              每个源: [stage1_feat(128x128)]
        Returns:
            fused_shallow: (B, out_channels, H, W)
        """
        max_feats = []
        for level in range(len(self.shallow_channels_list)):
            level_feats = torch.stack([f[level] for f in shallow_features], dim=0)
            max_feat, _ = level_feats.max(dim=0)
            max_feats.append(max_feat)
        concat = torch.cat(max_feats, dim=1)
        return self.project(concat)


# ========== V4 DecisionNet wrapper ==========
class DecisionNetV4(nn.Module):
    """V4 DecisionNet: 原始梯度差异 + 浅层纹理特征"""

    def __init__(self, num_source_images=5, base_channels=32, shallow_channels=32):
        super().__init__()
        self.decision_net = DecisionNet(
            num_source_images=num_source_images,
            num_scales=3,
            base_channels=base_channels,
        )
        # 融合浅层纹理特征
        self.shallow_fusion = nn.Sequential(
            nn.Conv2d(shallow_channels + base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, source_images, decoder_feat=None,
                decoder_proj=None, decoder_gate=None, shallow_feat=None):
        diff_feats = self.decision_net._compute_gradient_diff_feats(source_images)
        feat = self.decision_net.grad_ops(diff_feats)

        # 融合 decoder 特征
        if decoder_feat is not None and decoder_proj is not None:
            dec_feat = decoder_proj(decoder_feat)
            if dec_feat.shape[2:] != feat.shape[2:]:
                dec_feat = F.interpolate(dec_feat, size=feat.shape[2:], mode='bilinear', align_corners=False)
            if decoder_gate is not None:
                gate = decoder_gate(decoder_feat)
                if gate.shape[2:] != feat.shape[2:]:
                    gate = F.interpolate(gate, size=feat.shape[2:], mode='bilinear', align_corners=False)
                if gate.shape[1] == 1 and feat.shape[1] != 1:
                    gate = gate.expand(-1, feat.shape[1], -1, -1)
                feat = gate * feat + (1.0 - gate) * dec_feat
            else:
                feat = feat + dec_feat

        # 融合浅层纹理特征（上采样到相同分辨率）
        if shallow_feat is not None:
            if shallow_feat.shape[2:] != feat.shape[2:]:
                shallow_feat = F.interpolate(shallow_feat, size=feat.shape[2:],
                                              mode='bilinear', align_corners=False)
            feat = torch.cat([feat, shallow_feat], dim=1)
            feat = self.shallow_fusion(feat)

        logits = self.decision_net.out_conv(feat)
        return logits


# ========== V4 GumbelDecisionFusion ==========
class GumbelDecisionFusionV4(nn.Module):
    """V4 决策融合头 — 使用 DecisionNetV4 + 浅层跳跃"""

    def __init__(self, num_source_images=5, base_channels=32, gumbel_tau=0.67,
                 decoder_feat_channels=8, top_k=1,
                 gap_mix_enabled=False, gap_mix_threshold=0.15, gap_mix_alpha=0.9,
                 mode_refine_enabled=False, mode_refine_threshold=0.15, mode_refine_kernel_size=3,
                 bilateral_refine_enabled=False, bilateral_kernel_size=5,
                 bilateral_sigma_spatial=2.0, bilateral_sigma_color=0.1,
                 use_coarse_prior=False, coarse_prior_strength=0.4,
                 shallow_channels=32):
        super().__init__()
        self.num_source_images = num_source_images
        self.gumbel_tau = gumbel_tau
        self.top_k = top_k
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

        self.decision_net = DecisionNetV4(
            num_source_images=num_source_images,
            base_channels=base_channels,
            shallow_channels=shallow_channels,
        )
        self.shallow_skip = ShallowSkipToDecision(
            shallow_channels_list=[48],  # stage1 at 128x128
            out_channels=shallow_channels,
        )
        self.decoder_proj = nn.Sequential(
            nn.Conv2d(decoder_feat_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder_gate = nn.Sequential(
            nn.Conv2d(decoder_feat_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1, bias=True),
            nn.Sigmoid()
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

    def forward(self, decoder_features, source_images, shallow_features_list=None,
                coarse_prior_logits=None):
        # 计算浅层跳跃特征
        shallow_feat = None
        if shallow_features_list is not None:
            shallow_feat = self.shallow_skip(shallow_features_list)

        logits = self.decision_net(
            source_images,
            decoder_feat=decoder_features,
            decoder_proj=self.decoder_proj,
            decoder_gate=self.decoder_gate,
            shallow_feat=shallow_feat,
        )
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
                decision_map = gumbel_softmax_hard(logits, tau=self.gumbel_tau, dim=1)
                decision_map = decision_map.unsqueeze(2)
                fused = select_and_fuse(source_images, decision_map)
            else:
                if self.mode_refine_enabled:
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


# ========== V4 Main Model ==========
class MSegNetV4(nn.Module):
    """m-SegNet V4 — 浅层跳跃决策网络"""

    def __init__(self, num_source_images=5, in_channels=3,
                 stem_channels=24, stage_channels=[24, 48, 96, 128],
                 stage_blocks=[2, 4, 6, 3],
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
                 coarse_prior_hidden_channels=32, shallow_channels=32):
        super().__init__()
        self.num_source_images = num_source_images
        self.multi_source_bifpn_fusion = multi_source_bifpn_fusion
        self.top_k = top_k
        self.bifpn_out_channels = bifpn_out_channels
        self.bifpn_num_layers = bifpn_num_layers
        self.cross_source_alpha = cross_source_alpha

        self.encoder = LightEncoderV4(in_channels, stem_channels, stage_channels, stage_blocks)
        self.sppf = SPPF(in_channels=stage_channels[-1], out_channels=stage_channels[-1])
        self.use_bifpn = use_bifpn; self.use_simam = use_simam
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
        self.coarse_prior = SourceAwareCoarsePriorV4(
            in_channels=stage_channels[-1], hidden_channels=coarse_prior_hidden_channels,
        ) if use_coarse_prior else None
        self.decoder = LightDecoderV4(
            encoder_channels=stage_channels[::-1] + [stem_channels],
            decoder_channels=decoder_channels, bifpn_channels=bifpn_out_channels,
        )
        if use_fusion_head == 'gumbel':
            self.fusion_head = GumbelDecisionFusionV4(
                num_source_images=num_source_images, base_channels=32, gumbel_tau=0.67,
                decoder_feat_channels=decoder_tail_channels, top_k=self.top_k,
                gap_mix_enabled=gap_mix_enabled, gap_mix_threshold=gap_mix_threshold,
                gap_mix_alpha=gap_mix_alpha, mode_refine_enabled=mode_refine_enabled,
                mode_refine_threshold=mode_refine_threshold, mode_refine_kernel_size=mode_refine_kernel_size,
                bilateral_refine_enabled=bilateral_refine_enabled,
                bilateral_kernel_size=bilateral_kernel_size,
                bilateral_sigma_spatial=bilateral_sigma_spatial,
                bilateral_sigma_color=bilateral_sigma_color,
                use_coarse_prior=use_coarse_prior, coarse_prior_strength=coarse_prior_strength,
                shallow_channels=shallow_channels,
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
        num_stage_levels = len(encoded_features[0]['features']) - 1
        for level in range(num_stage_levels):
            level_feats = [enc['features'][level + 1] for enc in encoded_features]
            stacked = torch.stack(level_feats, dim=0)
            fused_level = stacked.mean(dim=0) if self.multi_source_bifpn_fusion == 'mean' else stacked.max(dim=0)[0]
            stage_outputs.append(fused_level)
        return stage_outputs

    def forward(self, source_images):
        encoded_features = []
        for src in source_images:
            feat = self.encoder(src)
            encoded_features.append(feat)
        # 收集浅层特征（stem + stage1，供 DecisionNet 使用）
        shallow_features_list = []
        for f in encoded_features:
            shallow_features_list.append([f['features'][2]])  # stage1 (128x128, 48ch)
        # Cross-source enhancement
        num_features = len(encoded_features[0]['features'])
        for level in range(num_features):
            level_feats = [f['features'][level] for f in encoded_features]
            stacked = torch.stack(level_feats, dim=0)
            max_feat, _ = stacked.max(dim=0, keepdim=True)
            for f in encoded_features:
                f['features'][level] = f['features'][level] + self.cross_source_alpha * max_feat.squeeze(0)
        enc_outs = [f['out'] for f in encoded_features]
        stacked_out = torch.stack(enc_outs, dim=0)
        max_out, _ = stacked_out.max(dim=0, keepdim=True)
        for f in encoded_features:
            f['out'] = f['out'] + self.cross_source_alpha * max_out.squeeze(0)
        stage_outputs = self._fuse_stage_outputs_for_bifpn(encoded_features)
        sppf_out = self.sppf(stage_outputs[-1])
        if self.use_simam: sppf_out = self.simam(sppf_out)
        stage_outputs[-1] = sppf_out
        bifpn_features = self.bifpn(stage_outputs) if (self.use_bifpn and self.bifpn is not None) else stage_outputs
        concatenated = torch.cat([f['out'] for f in encoded_features], dim=1)
        fused_deep = self.fusion_conv(concatenated)
        avg_encoder_features = []
        for i in range(len(encoded_features[0]['features'])):
            avg_feat = torch.stack([f['features'][i] for f in encoded_features], dim=0).mean(dim=0)
            avg_encoder_features.append(avg_feat)
        decoded = self.decoder(fused_deep, avg_encoder_features, bifpn_features)
        coarse_prior_logits = None
        if self.coarse_prior is not None:
            deepest_features = [f['out'] for f in encoded_features]
            coarse_prior_logits = self.coarse_prior(deepest_features, target_size=decoded.shape[2:])
        return self.fusion_head(decoded, source_images,
                                shallow_features_list=shallow_features_list,
                                coarse_prior_logits=coarse_prior_logits)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_source_images=5, **kwargs):
    return MSegNetV4(num_source_images=num_source_images, **kwargs)


# Identical to V2 encoder/decoder/coarse prior
class LightEncoderV4(nn.Module):
    def __init__(self, in_channels=3, stem_channels=16, stage_channels=[16, 32, 64, 128], stage_blocks=[2, 3, 4, 3]):
        super().__init__()
        self.stem = Stem(in_channels, stem_channels)
        self.stages = nn.ModuleList()
        prev_channels = stem_channels
        for oc, nb in zip(stage_channels, stage_blocks):
            self.stages.append(Stage(prev_channels, oc, nb, stride=2))
            prev_channels = oc
        self.stage_channels = stage_channels

    def forward(self, x):
        features = []; x = self.stem(x); features.append(x)
        for stage in self.stages: x = stage(x); features.append(x)
        return {'out': x, 'features': features}


class LightDecoderV4(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_skip_connection=True, bifpn_channels=64, num_bifpn_features=4):
        super().__init__()
        self.use_skip_connection = use_skip_connection; self.bifpn_channels = bifpn_channels; self.num_bifpn_features = num_bifpn_features
        self.bifpn_fusion = nn.ModuleList([
            nn.Sequential(nn.Conv2d(bifpn_channels, dc, 1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True))
            for dc in decoder_channels[:num_bifpn_features]
        ])
        self.blocks = nn.ModuleList(); prev_channels = bifpn_channels
        for i, (ec, dc) in enumerate(zip(encoder_channels, decoder_channels)):
            hb = i < num_bifpn_features; hs = use_skip_connection and i < len(encoder_channels) - 1
            ic = prev_channels
            if hb: ic += dc
            if hs: ic += ec
            conv = nn.Sequential(
                nn.Conv2d(ic, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True),
                nn.Conv2d(dc, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True),
            )
            self.blocks.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), conv))
            prev_channels = dc
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


class SourceAwareCoarsePriorV4(nn.Module):
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
