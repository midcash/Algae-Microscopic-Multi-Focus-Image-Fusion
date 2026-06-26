"""
m-SegNet V2 - 瘦身版

关键改动：
1. BiFPN：1层 × 64通道（原 2层×128通道）
2. Decoder：通道减半 [128,64,32,16,8]（原 [256,128,64,32,16]）
3. Encoder 末层：128（原 256）
4. Decoder 块内从 2层 conv 改为 1层 conv（参数量减半）
5. 融合头 weight_generator 简化为单卷积

目标参数量：~1.5-1.8M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.depthwise_conv import Stem, Stage
from models.modules.sppf import SPPF
from models.modules.bifpn import BiFPN
from models.modules.simam import SimAM
from models.modules.decision_net import DecisionNet, gumbel_softmax_hard, select_and_fuse, _gradient_magnitude, bilateral_refine_decision
from models.modules.algae_decision import AlgaeAwareDecisionPrior


class LightEncoder(nn.Module):
    """
    轻量级编码器
    ------------
    将单张图像逐步下采样，提取多尺度特征图。
    构成：Stem（快速降采样） + 多个 Stage（若干深度可分离残差块）。

    参数：
        in_channels: 输入图像通道数，默认3  通道数是描述图像数据结构的重要概念。它表示每个像素点所需的数值数量，用于描述颜色或其他特性。而彩色图像通常使用三通道，即RGB模式，分别表示红、绿、蓝三种颜色，每种颜色的值范围通常为0到255。
        stem_channels: Stem输出通道数
        stage_channels: 各阶段的输出通道列表
        stage_blocks: 各阶段包含的基本块数量
    """
    def __init__(self, in_channels=3, stem_channels=16,
                 stage_channels=[16, 32, 64, 128],
                 stage_blocks=[2, 3, 4, 3]):
        super().__init__()
        # Stem 进行初步的下采样和通道扩展   感受野（Receptive Field）指的是特征图上某个神经元的输出受输入图像上多大区域影响。
        self.stem = Stem(in_channels, stem_channels) #初步卷积，将 3 通道图像快速转换为 16 通道的低层特征图。
        self.stages = nn.ModuleList()
        prev_channels = stem_channels
        # 逐阶段构建
        for out_channels, num_blocks in zip(stage_channels, stage_blocks):
            # 每个 Stage 会将特征图高宽减半，然后堆叠若干残差块（stride=2）
            stage = Stage(prev_channels, out_channels, num_blocks, stride=2)
            self.stages.append(stage)
            prev_channels = out_channels       # 下一阶段的输入通道
        self.stage_channels = stage_channels   # 保存通道配置，供外部使用

    def forward(self, x):
        """
        输入：x (B, 3, H, W)
        返回：
            dict:
                'out': 最深层的特征图 (B, C_deepest, H/2^S, W/2^S)
                'features': 列表，[stem特征, stage1特征, ..., stageS特征]
        """
        features = []
        x = self.stem(x)            # 经过 Stem
        features.append(x)          # 记录 stem 输出
        for stage in self.stages:
            x = stage(x)            # 逐阶段下采样 + 残差块
            features.append(x)      # 记录每个 stage 输出
        return {'out': x, 'features': features}


class LightDecoder(nn.Module):
    """
    轻量级解码器
    ------------
    将编码器的最深层特征逐步上采样，同时融合多尺度跳跃连接和BiFPN特征。
    上采样采用双线性插值，每个解码块由两个卷积层组成（保留足够容量）。

    参数：
        encoder_channels: 编码器各阶段输出通道（反向顺序，深层到浅层）
        decoder_channels: 解码器各块输出通道列表
        use_skip_connection: 是否使用编码器跳跃连接
        bifpn_channels: BiFPN输出的通道数
        num_bifpn_features: BiFPN提供的多尺度特征数量（用于前几个解码块）
    """
    def __init__(self, encoder_channels, decoder_channels,
                 use_skip_connection=True, bifpn_channels=64, num_bifpn_features=4):
        super().__init__()
        self.use_skip_connection = use_skip_connection
        self.bifpn_channels = bifpn_channels
        self.num_bifpn_features = num_bifpn_features

        # 将 BiFPN 各层特征投影到对应解码块的通道维度
        self.bifpn_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bifpn_channels, dec_ch, 1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU6(inplace=True)
            ) for dec_ch in decoder_channels[:num_bifpn_features]
        ])

        self.blocks = nn.ModuleList()
        prev_channels = bifpn_channels  # 初始输入为融合后的最深层特征（通道数 = bifpn_channels）

        # 构建解码块：上采样 → 融合跳跃连接/BiFPN → 卷积块
        for i, (enc_ch, dec_ch) in enumerate(zip(encoder_channels, decoder_channels)):
            has_bifpn = i < num_bifpn_features
            has_skip = use_skip_connection and i < len(encoder_channels) - 1

            # 计算该解码块的输入通道数（包括前一层输出、BiFPN 投影特征、编码器跳跃连接）
            in_channels = prev_channels
            if has_bifpn:
                in_channels += dec_ch
            if has_skip:
                in_channels += enc_ch

            # 解码块：上采样 + 双卷积层（保持容量）
            conv = nn.Sequential(
                nn.Conv2d(in_channels, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(dec_ch, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU6(inplace=True)
            )

            self.blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                conv
            ))
            prev_channels = dec_ch

        # 记录每个解码块是否使用 BiFPN / 跳跃连接（用于 forward 中的条件判断）
        self.has_bifpn = [i < num_bifpn_features for i in range(len(decoder_channels))]
        self.has_skip = [
            use_skip_connection and i < len(encoder_channels) - 1
            for i in range(len(decoder_channels))
        ]

    def forward(self, x, encoder_features=None, bifpn_features=None):
        """
       输入：
           x: 融合后的最深层特征 (B, 64, H/16, W/16)
           encoder_features: 编码器的多尺度特征列表（用于跳跃连接）
           bifpn_features: BiFPN输出的多尺度特征列表
       返回：解码后的特征图 (B, decoder_tail_channels, H, W)
       """
        out = x
        for i, block in enumerate(self.blocks):
            inputs = [out]

            # 融合 BiFPN 特征（若存在且该解码块需要）
            if bifpn_features is not None and self.has_bifpn[i] and i < len(bifpn_features):
                bifpn_feat = F.interpolate(bifpn_features[i], size=out.shape[2:],
                                           mode='bilinear', align_corners=False)
                bifpn_feat = self.bifpn_fusion[i](bifpn_feat)
                inputs.append(bifpn_feat)

            # 融合编码器跳跃连接（若启用且该解码块需要）
            if self.use_skip_connection and encoder_features is not None:
                if self.has_skip[i] and i < len(encoder_features):
                    enc_feat = encoder_features[-(i+1)]         # 从深层到浅层取对应特征
                    if enc_feat.shape[2:] != out.shape[2:]:
                        enc_feat = F.interpolate(enc_feat, size=out.shape[2:],
                                                 mode='bilinear', align_corners=False)
                    inputs.append(enc_feat)

            # 拼接所有来源的特征，然后经过上采样和卷积块
            out = torch.cat(inputs, dim=1)
            out = block(out)
        return out


class SourceAwareCoarsePrior(nn.Module):
    """
    R21-min: source-aware coarse decision prior.

    对每张源图的最深层特征单独打分，保留源图身份，
    在低分辨率上产生 coarse logits，再上采样为决策偏置。
    """
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )

    def forward(self, deepest_features, target_size=None):
        # deepest_features: List[(B,C,h,w)]，每张源图各自保留
        coarse_scores = [self.score_head(feat) for feat in deepest_features]
        coarse_logits = torch.cat(coarse_scores, dim=1)  # (B, N, h, w)
        if target_size is not None and coarse_logits.shape[2:] != target_size:
            coarse_logits = F.interpolate(coarse_logits, size=target_size,
                                          mode='bilinear', align_corners=False)
        return coarse_logits


class GumbelDecisionFusion(nn.Module):
    """
    R15: Gumbel Softmax 决策融合头 + 解码器特征融合

    R18: 支持 top_k 加权融合。训练时仍用 argmax(Gumbel)，验证/推理时可用
    top_k 加权混合（每个像素融合 top-k 个源的边缘信息，提升 QABF）。

    forward 返回 (fused, decision_map, logits) 三件套


    Gumbel Softmax 决策融合头
    -------------------------
    通过 DecisionNet 预测每个像素的最优源图像索引，并支持 Top‑K 软融合。
    训练时注入 Gumbel 噪声，推理时使用 Top‑K 加权混合保留多个源的边缘细节。

    参数：
    num_source_images: 源图像数量
    base_channels: DecisionNet 内部基本通道数
    gumbel_tau: Gumbel softmax 温度
    decoder_feat_channels: 解码器输出特征通道数（默认与 decoder_tail_channels 一致）
    top_k: 推理时保留的 Top‑K 个源
    """
    def __init__(self, num_source_images=5, base_channels=32, gumbel_tau=0.67,
                 decoder_feat_channels=8, top_k=1,
                 gap_mix_enabled=False, gap_mix_threshold=0.15, gap_mix_alpha=0.9,
                 mode_refine_enabled=False, mode_refine_threshold=0.15, mode_refine_kernel_size=3,
                 bilateral_refine_enabled=False, bilateral_kernel_size=5,
                 bilateral_sigma_spatial=2.0, bilateral_sigma_color=0.1,
                 use_coarse_prior=False, coarse_prior_strength=0.4,
                 fusion_mode='gumbel'):
        super().__init__()
        self.num_source_images = num_source_images
        self.gumbel_tau = gumbel_tau
        self.top_k = top_k  # R18: 推理时保留 top-k 个源的边缘
        self.fusion_mode = fusion_mode  # 'gumbel' 或 'softmax'（消融用）
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

        # DecisionNet 用于从多源图像和解码器特征中预测每个像素的源索引 logits
        self.decision_net = DecisionNet(
            num_source_images=num_source_images,
            num_scales=3,
            base_channels=base_channels
        )
        # R15/V5-min: decoder_features 投影到 DecisionNet 内部特征空间
        self.decoder_proj = nn.Sequential(
            nn.Conv2d(decoder_feat_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # V5-min: 用 decoder 上下文生成单通道 gate，控制梯度分支与上下文分支的融合比例
        self.decoder_gate = nn.Sequential(
            nn.Conv2d(decoder_feat_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def _top_k_fuse(self, logits, source_images):
        """
        R18: Top-K 边缘加权融合

        训练/推理统一使用 top-k 加权混合。训练时 Gumbel 噪声仍用于计算 logits 梯度，
        但融合用软权重而非硬 argmax。

        Args:
            logits: (B, N, H, W) DecisionNet 输出
            source_images: List[Tensor] (N, B, 3, H, W)
        Returns:
            fused: (B, 3, H, W)
            weight_map: (B, N, H, W) — 每个源的融合权重（top-k 外为 0）
        """
        N = len(source_images)
        B, _, H, W = source_images[0].shape

        # 1. softmax 得到概率分布
        probs = F.softmax(logits, dim=1)  # (B, N, H, W)

        # 2. 梯度幅值加权的权重 = softmax_probs * grad_mag^0.3   计算每个源图像的梯度幅值，作为边缘显著性权重
        grads = []
        for src in source_images:
            g = _gradient_magnitude(src)    # 近似梯度幅值
            grads.append(g)
        grad_stack = torch.stack(grads, dim=1).squeeze(2)    # (B, N, H, W)

        # 权重 = softmax 概率 * (梯度幅值)^0.3，强调边缘区域
        weights = probs * (grad_stack + 1e-6).pow(0.3)

        # 3. top-k 掩码：保留权重最大的 k 个源
        k = min(self.top_k, N)
        # 找到第 k 大的权重值作为阈值
        thresh = torch.kthvalue(weights, k=N - k + 1, dim=1, keepdim=True)[0]  # 第 k 大的值
        mask = (weights >= thresh).float()

        # 4. top-k 内权重归一化
        weight_map = weights * mask
        weight_map = weight_map / (weight_map.sum(dim=1, keepdim=True) + 1e-8)

        # 5. 加权融合
        fused = torch.zeros_like(source_images[0])
        for i in range(N):
            w = weight_map[:, i:i+1].unsqueeze(2).expand(-1, -1, 3, -1, -1)
            fused += w[:, 0] * source_images[i]

        return fused, weight_map

    def _gap_mix_fuse(self, logits, source_images):
        """
        仅在推理阶段启用的 Top-2 Gap 弱混合。
        高置信区域保持 hard argmax；低置信区域在 top-1 与 top-2 之间做轻量软过渡，
        用于缓和边缘密集区的局部过锐化现象。
        """
        probs = F.softmax(logits, dim=1)  # (B, N, H, W)
        top2_vals, top2_idx = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)

        top1_val = top2_vals[:, 0:1]
        if top2_vals.shape[1] == 1:
            top2_val = torch.zeros_like(top1_val)
            top2_index = top2_idx[:, 0:1]
        else:
            top2_val = top2_vals[:, 1:2]
            top2_index = top2_idx[:, 1:2]
        top1_index = top2_idx[:, 0:1]

        gap = top1_val - top2_val
        low_conf_mask = (gap < self.gap_mix_threshold).float()  # (B,1,H,W)

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
        """
        R20-v1: 仅对低置信区域做局部多数投票平滑。
        低置信由 top-1/top-2 概率差判定；高置信区域保持原始硬决策不变。
        """
        probs = F.softmax(logits, dim=1)
        top2_vals, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
        top1 = top2_vals[:, 0:1]
        top2 = top2_vals[:, 1:2] if top2_vals.shape[1] > 1 else torch.zeros_like(top1)
        gap = top1 - top2  # (B,1,H,W)
        low_conf = gap < self.mode_refine_threshold

        base_idx = probs.argmax(dim=1, keepdim=True)  # (B,1,H,W)
        k = max(1, int(self.mode_refine_kernel_size))
        if k % 2 == 0:
            k += 1
        onehot = F.one_hot(base_idx.squeeze(1), num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        vote_counts = F.avg_pool2d(onehot, kernel_size=k, stride=1, padding=k // 2) * (k * k)
        refined_idx = vote_counts.argmax(dim=1, keepdim=True)
        final_idx = torch.where(low_conf, refined_idx, base_idx)
        decision_map = torch.zeros_like(probs).scatter_(1, final_idx, 1.0).unsqueeze(2)
        return decision_map, gap

    def forward(self, decoder_features, source_images, coarse_prior_logits=None):
        """
        R18: 训练/推理统一 top-k 融合

        Args:
            decoder_features: 解码器输出 (B, C=decoder_tail_channels, H, W)
            source_images: List[Tensor] (N, B, 3, H, W)
        Returns:
            fused: (B, 3, H, W)
            decision_map: (B, N, H, W) — top-k 软权重（训练）或 one-hot（top_k=1 或推理）
            logits: (B, N, H, W)

        输入：
            decoder_features: 解码器输出 (B, decoder_tail_channels, H, W)
            source_images: 源图像列表，每个 (B, 3, H, W)
            返回：
            fused: 融合后的图像 (B, 3, H, W)
            decision_map: 决策权重 (B, N, 1, H, W) （为了兼容损失函数）
            logits: DecisionNet 原始输出 (B, N, H, W)
            decoder_features: 原样返回解码器特征（供外部损失使用）
        """
        # 1. 计算 logits，解码器特征作为额外输入指导选择
        logits = self.decision_net(source_images, decoder_feat=decoder_features,
                                    decoder_proj=self.decoder_proj,
                                    decoder_gate=self.decoder_gate)  # (B, N, H, W)
        if self.use_coarse_prior and coarse_prior_logits is not None:
            logits = logits + self.coarse_prior_strength * coarse_prior_logits
        if self.top_k > 1:
            # 训练时添加 Gumbel 噪声增加探索
            # R18: top-k 融合，训练/推理统一
            if self.training:
                # 训练：用 Gumbel softmax 添加随机性
                uniform = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-10)
                gumbel = -torch.log(-torch.log(uniform))
                noisy_logits = (logits + gumbel) / self.gumbel_tau
            else:
                noisy_logits = logits

            fused, weight_map = self._top_k_fuse(noisy_logits, source_images)

            # decision_map 返回软权重（用于 loss 计算中的 GradContrast 等）
            decision_map = weight_map  # (B, N, H, W)，与旧版 (B,N,1,H,W) 不同！

            # 为使 loss 兼容，unsqueeze 到 (B, N, 1, H, W)
            decision_map = decision_map.unsqueeze(2)
        else:
            # top_k=1 = 标准 argmax（R16 兼容模式）
            if self.training:
                if self.fusion_mode == 'softmax':
                    # 消融：Plain Softmax 加权混合（无 Gumbel，无 STE）
                    probs = F.softmax(logits, dim=1)  # (B, N, H, W)
                    decision_map = probs.unsqueeze(2)  # (B, N, 1, H, W)
                    fused = torch.zeros_like(source_images[0])
                    for k in range(self.num_source_images):
                        fused += probs[:, k:k+1] * source_images[k]
                else:
                    decision_map = gumbel_softmax_hard(logits, tau=self.gumbel_tau, dim=1)
                    decision_map = decision_map.unsqueeze(2)
                    fused = select_and_fuse(source_images, decision_map)
            else:
                if self.fusion_mode == 'softmax':
                    # 消融推理：Plain Softmax 加权混合
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
                    # R22: 双边精炼硬决策 — 在同质区域平滑碎片，在结构边界保持硬选择
                    idx = logits.argmax(dim=1, keepdim=True)
                    raw_decision = torch.zeros_like(logits).scatter_(1, idx, 1.0).unsqueeze(2)
                    # 用任意源图作为引导（取第一张），提供结构信息
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


class MSegNetV2(nn.Module):
    """
     m‑SegNet V2 —— 瘦身版多聚焦图像融合网络
     ------------------------------------------
     结构概览：
     1. LightEncoder: 对每张源图像编码，并在各阶段做跨源特征增强
     2. SPPF + SimAM: 对最深层特征进行多尺度池化和注意力增强
     3. BiFPN: 多尺度双向特征融合
     4. LightDecoder: 逐步上采样，恢复高分辨率特征
     5. GumbelDecisionFusion: 逐像素决策，加权融合多源图像

     目标参数量：约 1.5M ~ 1.8M
     """
    def __init__(self, num_source_images=5, in_channels=3,
                 stem_channels=24,
                 stage_channels=[24, 48, 96, 128],
                 stage_blocks=[2, 4, 6, 3],
                 use_bifpn=True,
                 use_simam=True,
                 use_fusion_head='decision',
                 multi_source_bifpn_fusion='mean',
                 bifpn_out_channels=64,
                 bifpn_num_layers=2,
                 decoder_tail_channels=8,
                 cross_source_alpha=0.2,
                 top_k=1,
                 gap_mix_enabled=False,
                 gap_mix_threshold=0.15,
                 gap_mix_alpha=0.9,
                 mode_refine_enabled=False,
                 mode_refine_threshold=0.15,
                 mode_refine_kernel_size=3,
                 bilateral_refine_enabled=False,
                 bilateral_kernel_size=5,
                 bilateral_sigma_spatial=2.0,
                 bilateral_sigma_color=0.1,
                 use_coarse_prior=False,
                 coarse_prior_strength=0.4,
                 coarse_prior_hidden_channels=32,
                 fusion_mode='gumbel'):  # 'gumbel' 或 'softmax'（消融：Gumbel→Softmax）
        super().__init__()

        self.num_source_images = num_source_images
        self.multi_source_bifpn_fusion = multi_source_bifpn_fusion  # 'first' 或 'mean'/'max'
        self.top_k = top_k
        self.bifpn_out_channels = bifpn_out_channels
        self.bifpn_num_layers = bifpn_num_layers
        self.cross_source_alpha = cross_source_alpha
        self.use_coarse_prior = use_coarse_prior
        self.coarse_prior_strength = coarse_prior_strength
        self.coarse_prior_hidden_channels = coarse_prior_hidden_channels

        # 编码器（轻量）
        self.encoder = LightEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks
        )

        # 多尺度池化模块（扩大感受野，不改变尺寸）
        self.sppf = SPPF(in_channels=stage_channels[-1], out_channels=stage_channels[-1])

        self.use_bifpn = use_bifpn
        self.use_simam = use_simam

        # 双向特征金字塔网络（多尺度特征融合）
        self.bifpn = BiFPN(
            in_channels_list=stage_channels,
            out_channels=bifpn_out_channels,
            num_levels=len(stage_channels),
            num_layers=bifpn_num_layers
        ) if use_bifpn else None

        # 空间注意力模块
        if use_simam:
            self.simam = SimAM()

        # 融合 conv：将多个源的最深层特征拼接后投影到 64 通道，作为解码器输入
        fusion_input_channels = stage_channels[-1] * num_source_images
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_input_channels, bifpn_out_channels, 1, bias=False),
            nn.BatchNorm2d(bifpn_out_channels),
            nn.ReLU6(inplace=True)
        )

        # 解码器通道配置（从深层到浅层，最后输出 decoder_tail_channels 通道）
        decoder_channels = [128, 64, 32, 16, decoder_tail_channels]
        self.coarse_prior = SourceAwareCoarsePrior(
            in_channels=stage_channels[-1],
            hidden_channels=coarse_prior_hidden_channels
        ) if use_coarse_prior else None
        self.decoder = LightDecoder(
            encoder_channels=stage_channels[::-1] + [stem_channels], # 包含 stem 输出
            decoder_channels=decoder_channels,
            bifpn_channels=bifpn_out_channels
        )

        # R15+: 只使用 GumbelDecisionFusion（LightDecisionMapFusion 已被废弃）
        if use_fusion_head == 'gumbel':
            # R11: Gumbel Decision Fusion
            # R18: top_k 参数 — 推理时保留 top-k 个源的边缘
            self.fusion_head = GumbelDecisionFusion(
                num_source_images=num_source_images,
                base_channels=32,
                gumbel_tau=0.67,
                decoder_feat_channels=decoder_tail_channels,
                top_k=self.top_k,
                gap_mix_enabled=gap_mix_enabled,
                gap_mix_threshold=gap_mix_threshold,
                gap_mix_alpha=gap_mix_alpha,
                mode_refine_enabled=mode_refine_enabled,
                mode_refine_threshold=mode_refine_threshold,
                mode_refine_kernel_size=mode_refine_kernel_size,
                bilateral_refine_enabled=bilateral_refine_enabled,
                bilateral_kernel_size=bilateral_kernel_size,
                bilateral_sigma_spatial=bilateral_sigma_spatial,
                bilateral_sigma_color=bilateral_sigma_color,
                use_coarse_prior=use_coarse_prior,
                coarse_prior_strength=coarse_prior_strength,
                fusion_mode=fusion_mode
            )
        else:
            # 回退：没有 fusion head 时直接返回 decoder 输出（基本不用）
            self.fusion_head = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming 初始化卷积层，BN 层初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _fuse_stage_outputs_for_bifpn(self, encoded_features):
        """
       将多源图像编码器在各阶段的特征进行融合，得到单一的多尺度特征列表供 BiFPN 使用。

       Args:
           encoded_features: List[dict]，每个 dict 包含 'features' 列表（如 [stem, stage1, ...]）
       Returns:
           stage_outputs: List[Tensor]，每个元素对应一个阶段（从 stage1 开始）的融合特征
       """
        if self.multi_source_bifpn_fusion == 'first':
            # 简单取第一个源的各阶段特征（仅用于调试）
            return list(encoded_features[0]['features'][1:])
        stage_outputs = []
        num_stage_levels = len(encoded_features[0]['features']) - 1  # 不包括 stem

        for level in range(num_stage_levels):
            # 收集所有源图在该阶段的特征
            level_feats = [enc['features'][level + 1] for enc in encoded_features]
            stacked = torch.stack(level_feats, dim=0)
            if self.multi_source_bifpn_fusion == 'mean':
                fused_level = stacked.mean(dim=0)     # 平均融合
            else:
                fused_level = stacked.max(dim=0)[0]   # 最大融合（默认）
            stage_outputs.append(fused_level)
        return stage_outputs

    def forward(self, source_images):
        """
        前向传播：输入多张源图像，输出融合结果及中间信息。

        Args:
            source_images: List[Tensor]，长度为 N，每个 Tensor 形状 (B, 3, H, W)
        Returns:
            fused: (B, 3, H, W) 融合图像
            decision_map: (B, N, 1, H, W) 决策权重图（用于损失计算）
            logits: (B, N, H, W) DecisionNet 原始 logits
            decoder_features: (B, C, H, W) 解码器输出特征（用于特征级损失）
        """
        # ------------------- 编码阶段 -------------------
        # 编码阶段 + 跨源特征聚合（每个 stage 后共享焦点信息）
        encoded_features = []  # 存储每个源的 encoder 输出（包含 'out' 和 'features'）
        batch_size = source_images[0].shape[0]

        # 逐源图编码，在每个 stage 后做跨源聚合
        for src_idx, src in enumerate(source_images):
            feat = self.encoder(src)
            encoded_features.append(feat)

        # ------------------- 跨源特征增强（channel-wise max） -------------------
        # 在每个 stage 的特征上，取所有源图对应 stage 的最大响应，并以残差方式增强每个源的特征
        num_features = len(encoded_features[0]['features'])
        for level in range(num_features):
            # 收集所有源图在该 stage 的特征
            level_feats = [f['features'][level] for f in encoded_features]
            # channel-wise max = 取各源图该通道的最大响应
            # 在多焦距场景中，响应最大的通道通常对应最清晰的区域
            stacked = torch.stack(level_feats, dim=0)  # (N, B, C, H, W)
            max_feat, _ = stacked.max(dim=0, keepdim=True)  # (1, B, C, H, W)
            # 用 max 特征增强各源图（残差方式，保留原来信息）
            for f_idx, f in enumerate(encoded_features):
                f['features'][level] = f['features'][level] + self.cross_source_alpha * max_feat.squeeze(0)

        # 编码器最终输出也用跨源 max 增强
        enc_outs = [f['out'] for f in encoded_features]
        stacked_out = torch.stack(enc_outs, dim=0)
        max_out, _ = stacked_out.max(dim=0, keepdim=True)
        for f_idx, f in enumerate(encoded_features):
            f['out'] = f['out'] + self.cross_source_alpha * max_out.squeeze(0)
        # ------------------- 融合多源特征供 BiFPN 使用 -------------------
        stage_outputs = self._fuse_stage_outputs_for_bifpn(encoded_features)

        # ------------------- 最深层特征增强（SPPF + SimAM） -----------------
        sppf_out = self.sppf(stage_outputs[-1])
        if self.use_simam:
            sppf_out = self.simam(sppf_out)
        stage_outputs[-1] = sppf_out

        # ------------------- BiFPN 多尺度融合 -------------------
        if self.use_bifpn and self.bifpn is not None:
            bifpn_features = self.bifpn(stage_outputs)  # 列表，每个元素形状 (B,64,H_i,W_i)
        else:
            bifpn_features = stage_outputs

        # ------------------- 融合最深特征作为解码器输入 -------------------
        # 将所有源的最深层特征（已增强）在通道维度拼接，然后投影到 64 通道
        concatenated = torch.cat([f['out'] for f in encoded_features], dim=1)
        fused = self.fusion_conv(concatenated)

        # ------------------- 准备编码器跳跃连接（平均融合） -------------------
        # 对每个 stage 的特征，取所有源的平均值作为跳跃连接（避免信息偏向单一源）
        avg_encoder_features = []
        for i in range(len(encoded_features[0]['features'])):
            avg_feat = torch.stack([f['features'][i] for f in encoded_features], dim=0).mean(dim=0)
            avg_encoder_features.append(avg_feat)

        # ------------------- 解码器 -------------------
        decoded = self.decoder(fused, avg_encoder_features, bifpn_features)
        coarse_prior_logits = None
        if self.coarse_prior is not None:
            deepest_features = [f['out'] for f in encoded_features]
            coarse_prior_logits = self.coarse_prior(deepest_features, target_size=decoded.shape[2:])
        # 输出 4 元组 (fused, decision_map, logits, decoder_features)
        # decoder_features 传回给 CombinedLoss 做特征级一致性监督
        output = self.fusion_head(decoded, source_images, coarse_prior_logits=coarse_prior_logits)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_v2(num_source_images=5, **kwargs):
    return MSegNetV2(num_source_images=num_source_images, **kwargs)


# 为方便 train.py 调用，保留原接口
create_model = create_model_v2
