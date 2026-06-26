"""
m-SegNet V6 — Laplacian Oracle 引导 + 源图保真 + 解码器弱化

V5诊断结论:
- 伪标签训练的模型与Laplacian oracle不一致(仅21-26%)
- V5更自信但选错更多
- 模型选了Source 5最多(32%),而非真正的Source 2

V6方案:
1. 训练目标: 直接用Laplacian oracle作为训练信号(不依赖伪标签)
2. 源图保真损失: 高置信区域约束融合结果接近选中源图
3. 解码器弱化: decoder特征通过可学习gate控制贡献,初始化为低权重
4. 低置信一致: low-conf区域允许局部平滑
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.depthwise_conv import Stem, Stage
from models.modules.sppf import SPPF
from models.modules.bifpn import BiFPN
from models.modules.simam import SimAM
from models.modules.decision_net import (
    gumbel_softmax_hard, select_and_fuse, _gradient_magnitude,
    bilateral_refine_decision,
)


class DecisionNetV6(nn.Module):
    """V6 DecisionNet: 梯度差异 + raw Laplacian，解码器可选（弱化gate）"""

    def __init__(self, num_source_images=5, base_channels=32):
        super().__init__()
        N = num_source_images
        num_pairs = N * (N - 1) // 2
        grad_in = num_pairs * 3  # 3 scales
        raw_lap_in = N * 2  # 2 scales per source

        # 梯度差异分支
        self.grad_ops = nn.Sequential(
            nn.Conv2d(grad_in, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels, bias=False),
            nn.Conv2d(base_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )
        # Raw Laplacian分支
        self.raw_lap_proj = nn.Sequential(
            nn.Conv2d(raw_lap_in, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2), nn.ReLU(inplace=True),
        )
        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels + base_channels // 2, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(base_channels, N, 3, padding=1, bias=True)
        self.register_buffer('lap_k', torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32))

    def _compute_gradient_diff_feats(self, source_images):
        N = len(source_images); B, _, H, W = source_images[0].shape
        all_diff = []
        for scale in [1.0, 0.5, 0.25]:
            if scale < 1.0:
                sH, sW = int(H * scale), int(W * scale)
                scaled = [F.interpolate(s, size=(sH, sW), mode='bilinear', align_corners=False) for s in source_images]
            else:
                scaled = source_images
            grads = [_gradient_magnitude(s) for s in scaled]
            diffs = []
            for i in range(N):
                for j in range(i + 1, N):
                    diffs.append(torch.abs(grads[i] - grads[j]))
            feat = torch.cat(diffs, dim=1)
            if scale < 1.0:
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            all_diff.append(feat)
        return torch.cat(all_diff, dim=1)

    def _compute_raw_lap_feats(self, source_images):
        N = len(source_images); B, _, H, W = source_images[0].shape
        all_lap = []
        for scale in [1.0, 0.5]:
            for src in source_images:
                gray = src.mean(dim=1, keepdim=True)
                if scale < 1.0:
                    sH, sW = int(H * scale), int(W * scale)
                    gray = F.interpolate(gray, size=(sH, sW), mode='bilinear', align_corners=False)
                lap = F.conv2d(gray, self.lap_k, padding=1).abs()
                if scale < 1.0:
                    lap = F.interpolate(lap, size=(H, W), mode='bilinear', align_corners=False)
                all_lap.append(lap)
        return torch.cat(all_lap, dim=1)

    def forward(self, source_images, decoder_feat=None, decoder_proj=None, decoder_gate=None):
        grad_feat = self.grad_ops(self._compute_gradient_diff_feats(source_images))
        lap_feat = self.raw_lap_proj(self._compute_raw_lap_feats(source_images))
        feat = self.fusion(torch.cat([grad_feat, lap_feat], dim=1))

        # V6: 解码器特征以更低的初始权重参与（gate初始化为0.1偏置）
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
                # V6关键: decoder gate初始化为低值(通过bias=-2)
                feat = feat + gate * dec_feat  # 不再是(1-gate)*dec_feat
            else:
                feat = feat + 0.1 * dec_feat  # 固定低权重

        return self.out_conv(feat)


class GumbelDecisionFusionV6(nn.Module):
    """V6 决策融合头"""

    def __init__(self, num_source_images=5, base_channels=32, gumbel_tau=0.67,
                 decoder_feat_channels=8, top_k=1,
                 gap_mix_enabled=False, gap_mix_threshold=0.15, gap_mix_alpha=0.9,
                 use_coarse_prior=False, coarse_prior_strength=0.4):
        super().__init__()
        self.num_source_images = num_source_images
        self.gumbel_tau = gumbel_tau; self.top_k = top_k
        self.gap_mix_enabled = gap_mix_enabled; self.gap_mix_threshold = gap_mix_threshold
        self.gap_mix_alpha = gap_mix_alpha
        self.use_coarse_prior = use_coarse_prior; self.coarse_prior_strength = coarse_prior_strength

        self.decision_net = DecisionNetV6(num_source_images=num_source_images, base_channels=base_channels)
        self.decoder_proj = nn.Sequential(
            nn.Conv2d(decoder_feat_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))
        # V6: decoder_gate初始化为低值(sigmoid(-2)≈0.12)
        self.decoder_gate = nn.Sequential(
            nn.Conv2d(decoder_feat_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1, bias=True), nn.Sigmoid())
        # 初始化gate bias为负值
        nn.init.constant_(self.decoder_gate[3].bias, -2.0)

    def _gap_mix_fuse(self, logits, source_images):
        probs = F.softmax(logits, dim=1)
        top2_vals, top2_idx = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
        top1_val = top2_vals[:, 0:1]
        top2_val = top2_vals[:, 1:2] if top2_vals.shape[1] > 1 else torch.zeros_like(top1_val)
        top2_index = top2_idx[:, 1:2] if top2_idx.shape[1] > 1 else top2_idx[:, 0:1]
        top1_index = top2_idx[:, 0:1]
        gap = top1_val - top2_val
        low_conf = (gap < self.gap_mix_threshold).float()
        w1 = (1.0 - low_conf) + low_conf * self.gap_mix_alpha
        w2 = low_conf * (1.0 - self.gap_mix_alpha)
        wm = torch.zeros_like(probs)
        wm.scatter_(1, top1_index, w1); wm.scatter_add_(1, top2_index, w2)
        fused = torch.zeros_like(source_images[0])
        for i in range(len(source_images)):
            fused += wm[:, i:i+1] * source_images[i]
        return fused, wm.unsqueeze(2), gap

    def forward(self, decoder_features, source_images, coarse_prior_logits=None):
        logits = self.decision_net(source_images, decoder_feat=decoder_features,
                                    decoder_proj=self.decoder_proj, decoder_gate=self.decoder_gate)
        if self.use_coarse_prior and coarse_prior_logits is not None:
            logits = logits + self.coarse_prior_strength * coarse_prior_logits

        if self.training:
            decision_map = gumbel_softmax_hard(logits, tau=self.gumbel_tau, dim=1)
            decision_map = decision_map.unsqueeze(2)
            fused = select_and_fuse(source_images, decision_map)
        else:
            if self.gap_mix_enabled:
                fused, decision_map, _ = self._gap_mix_fuse(logits, source_images)
            else:
                idx = logits.argmax(dim=1, keepdim=True)
                decision_map = torch.zeros_like(logits).scatter_(1, idx, 1.0).unsqueeze(2)
                fused = select_and_fuse(source_images, decision_map)
        return fused, decision_map, logits, decoder_features


class MSegNetV6(nn.Module):
    """m-SegNet V6 — Laplacian Oracle引导 + 源图保真"""

    def __init__(self, num_source_images=5, in_channels=3,
                 stem_channels=24, stage_channels=None, stage_blocks=None,
                 use_bifpn=True, use_simam=True, use_fusion_head='decision',
                 multi_source_bifpn_fusion='mean',
                 bifpn_out_channels=64, bifpn_num_layers=2,
                 decoder_tail_channels=8, cross_source_alpha=0.2,
                 top_k=1, gap_mix_enabled=False, gap_mix_threshold=0.15,
                 gap_mix_alpha=0.9,
                 use_coarse_prior=False, coarse_prior_strength=0.4,
                 coarse_prior_hidden_channels=32):
        if stage_channels is None: stage_channels = [24, 48, 96, 128]
        if stage_blocks is None: stage_blocks = [2, 4, 6, 3]
        super().__init__()
        self.num_source_images = num_source_images
        self.multi_source_bifpn_fusion = multi_source_bifpn_fusion
        self.cross_source_alpha = cross_source_alpha

        self.encoder = LightEncoderV6(in_channels, stem_channels, stage_channels, stage_blocks)
        self.sppf = SPPF(in_channels=stage_channels[-1], out_channels=stage_channels[-1])
        self.use_bifpn = use_bifpn; self.use_simam = use_simam
        self.bifpn = BiFPN(in_channels_list=stage_channels, out_channels=bifpn_out_channels,
                           num_levels=len(stage_channels), num_layers=bifpn_num_layers) if use_bifpn else None
        if use_simam: self.simam = SimAM()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(stage_channels[-1] * num_source_images, bifpn_out_channels, 1, bias=False),
            nn.BatchNorm2d(bifpn_out_channels), nn.ReLU6(inplace=True))
        decoder_channels = [128, 64, 32, 16, decoder_tail_channels]
        self.coarse_prior = SourceAwareCoarsePriorV6(
            in_channels=stage_channels[-1], hidden_channels=coarse_prior_hidden_channels,
        ) if use_coarse_prior else None
        self.decoder = LightDecoderV6(
            encoder_channels=stage_channels[::-1] + [stem_channels],
            decoder_channels=decoder_channels, bifpn_channels=bifpn_out_channels)
        if use_fusion_head == 'gumbel':
            self.fusion_head = GumbelDecisionFusionV6(
                num_source_images=num_source_images, base_channels=32, gumbel_tau=0.67,
                decoder_feat_channels=decoder_tail_channels, top_k=top_k,
                gap_mix_enabled=gap_mix_enabled, gap_mix_threshold=gap_mix_threshold,
                gap_mix_alpha=gap_mix_alpha,
                use_coarse_prior=use_coarse_prior, coarse_prior_strength=coarse_prior_strength)
        else:
            self.fusion_head = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _fuse_for_bifpn(self, encoded_features):
        if self.multi_source_bifpn_fusion == 'first':
            return list(encoded_features[0]['features'][1:])
        outputs = []
        for lv in range(len(encoded_features[0]['features']) - 1):
            stacked = torch.stack([e['features'][lv + 1] for e in encoded_features], dim=0)
            outputs.append(stacked.mean(dim=0) if self.multi_source_bifpn_fusion == 'mean' else stacked.max(dim=0)[0])
        return outputs

    def forward(self, source_images):
        encoded = [self.encoder(s) for s in source_images]
        nf = len(encoded[0]['features'])
        for lv in range(nf):
            s = torch.stack([e['features'][lv] for e in encoded], dim=0)
            mx, _ = s.max(dim=0, keepdim=True)
            for e in encoded: e['features'][lv] = e['features'][lv] + self.cross_source_alpha * mx.squeeze(0)
        so = torch.stack([e['out'] for e in encoded], dim=0)
        mx, _ = so.max(dim=0, keepdim=True)
        for e in encoded: e['out'] = e['out'] + self.cross_source_alpha * mx.squeeze(0)
        stage_outs = self._fuse_for_bifpn(encoded)
        sp = self.sppf(stage_outs[-1])
        if self.use_simam: sp = self.simam(sp)
        stage_outs[-1] = sp
        bfpn = self.bifpn(stage_outs) if (self.use_bifpn and self.bifpn is not None) else stage_outs
        concat = torch.cat([e['out'] for e in encoded], dim=1)
        fd = self.fusion_conv(concat)
        avg_enc = [torch.stack([e['features'][i] for e in encoded], dim=0).mean(dim=0) for i in range(nf)]
        dec = self.decoder(fd, avg_enc, bfpn)
        cp = None
        if self.coarse_prior is not None:
            cp = self.coarse_prior([e['out'] for e in encoded], target_size=dec.shape[2:])
        return self.fusion_head(dec, source_images, cp)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_source_images=5, **kwargs):
    return MSegNetV6(num_source_images=num_source_images, **kwargs)


# Re-used encoder/decoder
class LightEncoderV6(nn.Module):
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


class LightDecoderV6(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_skip_connection=True,
                 bifpn_channels=64, num_bifpn_features=4):
        super().__init__()
        self.use_skip = use_skip_connection
        self.bf_fuse = nn.ModuleList([nn.Sequential(
            nn.Conv2d(bifpn_channels, dc, 1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True))
            for dc in decoder_channels[:num_bifpn_features]])
        self.blocks = nn.ModuleList(); pc = bifpn_channels
        for i, (ec, dc) in enumerate(zip(encoder_channels, decoder_channels)):
            hb = i < num_bifpn_features
            hs = use_skip_connection and i < len(encoder_channels) - 1
            ic = pc
            if hb: ic += dc
            if hs: ic += ec
            self.blocks.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Sequential(nn.Conv2d(ic, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True),
                              nn.Conv2d(dc, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU6(inplace=True))))
            pc = dc
        self.has_bf = [i < num_bifpn_features for i in range(len(decoder_channels))]
        self.has_sk = [use_skip_connection and i < len(encoder_channels) - 1 for i in range(len(decoder_channels))]

    def forward(self, x, encoder_features=None, bifpn_features=None):
        out = x
        for i, blk in enumerate(self.blocks):
            ins = [out]
            if bifpn_features is not None and self.has_bf[i] and i < len(bifpn_features):
                bf = F.interpolate(bifpn_features[i], size=out.shape[2:], mode='bilinear', align_corners=False)
                ins.append(self.bf_fuse[i](bf))
            if self.use_skip and encoder_features is not None and self.has_sk[i] and i < len(encoder_features):
                ef = encoder_features[-(i+1)]
                if ef.shape[2:] != out.shape[2:]: ef = F.interpolate(ef, size=out.shape[2:], mode='bilinear', align_corners=False)
                ins.append(ef)
            out = blk(torch.cat(ins, dim=1))
        return out


class SourceAwareCoarsePriorV6(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1, bias=True))

    def forward(self, deepest_features, target_size=None):
        cl = torch.cat([self.score_head(f) for f in deepest_features], dim=1)
        if target_size is not None and cl.shape[2:] != target_size:
            cl = F.interpolate(cl, size=target_size, mode='bilinear', align_corners=False)
        return cl
