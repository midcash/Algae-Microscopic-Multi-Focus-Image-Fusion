"""
m-SegNet 主模型

基于 SegNet 改进的轻量级多聚焦图像融合网络。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.depthwise_conv import DepthwiseConvBlock, Stem, Stage
from models.modules.sppf import SPPF
from models.modules.bifpn import BiFPN
from models.modules.simam import SimAM
from models.modules.algae_decision import AlgaeAwareDecisionPrior


class Encoder(nn.Module):
    """
    轻量级编码器

    使用 Depthwise Conv 构建，提取多尺度特征。

    Args:
        in_channels: 输入通道数 (默认 3)
        stem_channels: Stem 输出通道数
        stage_channels: 各阶段输出通道数列表
        stage_blocks: 各阶段块数量列表

    Returns:
        dict: {
            'out': 最终输出特征,
            'features': 各阶段特征列表 (用于跳跃连接)
        }
    """

    def __init__(self, in_channels=3, stem_channels=32,
                 stage_channels=[32, 64, 128, 256],
                 stage_blocks=[2, 4, 6, 4]):
        super().__init__()

        # Stem
        self.stem = Stem(in_channels, stem_channels)

        # 多阶段编码器
        self.stages = nn.ModuleList()
        prev_channels = stem_channels

        for out_channels, num_blocks in zip(stage_channels, stage_blocks):
            stage = Stage(prev_channels, out_channels, num_blocks, stride=2)
            self.stages.append(stage)
            prev_channels = out_channels

        self.stage_channels = stage_channels

    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            dict: {'out': Tensor, 'features': List[Tensor]}
                  features 包含 [stem, stage1, stage2, stage3, stage4] 共 5 层特征
        """
        features = []

        x = self.stem(x)
        features.append(x)  # stem 输出 (B, stem_channels, H/2, W/2)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return {
            'out': x,
            'features': features
        }


class Decoder(nn.Module):
    """
    轻量级解码器

    逐步上采样恢复空间分辨率，结合 BiFPN 特征和跳跃连接。

    Args:
        encoder_channels: 编码器各阶段通道数 (逆序)，长度 = 5 (stem + 4 stages)
        decoder_channels: 解码器各阶段通道数，长度 = 5
        use_skip_connection: 是否使用跳跃连接
        bifpn_channels: BiFPN 输出通道数
        num_bifpn_features: BiFPN 输出特征层数 (默认 4，不包含 stem)
    """

    def __init__(self, encoder_channels, decoder_channels,
                 use_skip_connection=True, bifpn_channels=128, num_bifpn_features=4):
        super().__init__()
        self.use_skip_connection = use_skip_connection
        self.bifpn_channels = bifpn_channels
        self.num_bifpn_features = num_bifpn_features

        # BiFPN 特征融合卷积 (将 BiFPN 特征投影到解码器通道)
        # 只为有 BiFPN 特征的 block 创建投影层
        self.bifpn_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bifpn_channels, dec_ch, 1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU6(inplace=True)
            )
            for dec_ch in decoder_channels[:num_bifpn_features]
        ])

        # 解码器块
        self.blocks = nn.ModuleList()
        # 第一个块的输入是 fusion_conv 输出 (128 通道)
        prev_channels = 128

        for i, (enc_ch, dec_ch) in enumerate(zip(encoder_channels, decoder_channels)):
            # 上采样
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

            # 判断当前 block 是否有 BiFPN 特征融合
            # BiFPN 只有 num_bifpn_features 层，对应前 num_bifpn_features 个 block
            has_bifpn = i < num_bifpn_features

            # 判断当前 block 是否有 skip connection
            # 最后一个 block (i=4) 没有 skip connection（浅层特征语义信息少）
            has_skip = use_skip_connection and i < len(encoder_channels) - 1

            # 计算输入通道数
            in_channels = prev_channels  # 上一层输出
            if has_bifpn:
                in_channels += dec_ch  # BiFPN 特征投影后
            if has_skip:
                in_channels += enc_ch  # 编码器跳跃连接

            # 卷积块
            conv = nn.Sequential(
                nn.Conv2d(in_channels, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(dec_ch, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU6(inplace=True)
            )

            self.blocks.append(nn.Sequential(upsample, conv))
            prev_channels = dec_ch

        # 保存各 block 的标志（用于 forward 保持一致）
        self.has_bifpn = [i < num_bifpn_features for i in range(len(decoder_channels))]
        self.has_skip = [
            use_skip_connection and i < len(encoder_channels) - 1
            for i in range(len(decoder_channels))
        ]

    def forward(self, x, encoder_features=None, bifpn_features=None):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            encoder_features: 编码器特征列表 (用于跳跃连接)，长度应为 5
            bifpn_features: BiFPN 特征列表 (用于多尺度融合)，长度应为 4

        Returns:
            Tensor: 解码输出
        """
        out = x

        for i, block in enumerate(self.blocks):
            inputs = [out]

            # 融合 BiFPN 特征 - 使用与__init__一致的条件
            if bifpn_features is not None and self.has_bifpn[i] and i < len(bifpn_features):
                # 调整 BiFPN 特征尺寸以匹配并投影
                bifpn_feat = F.interpolate(bifpn_features[i], size=out.shape[2:],
                                           mode='bilinear', align_corners=False)
                bifpn_feat = self.bifpn_fusion[i](bifpn_feat)
                inputs.append(bifpn_feat)

            # 跳跃连接 - 使用与 __init__ 一致的条件
            if self.use_skip_connection and encoder_features is not None:
                if self.has_skip[i] and i < len(encoder_features):
                    # encoder_features 顺序：[stem, stage1, stage2, stage3, stage4]
                    # 解码器从低分辨率到高分率，所以用 -(i+1) 取对应特征
                    enc_feat = encoder_features[-(i+1)]
                    # 调整尺寸匹配
                    if enc_feat.shape[2:] != out.shape[2:]:
                        enc_feat = F.interpolate(enc_feat, size=out.shape[2:],
                                                 mode='bilinear', align_corners=False)
                    inputs.append(enc_feat)

            # 拼接所有输入
            out = torch.cat(inputs, dim=1)

            # 应用解码块
            out = block(out)

        return out


class DecisionMapFusion(nn.Module):
    """
    决策图融合头

    生成每个源图的权重图，然后加权融合。

    Args:
        num_source_images: 源图像数量
        channels: 解码器输出通道数
        use_algae_decision_prior: 是否启用显式先验决策图
        decision_prior_mode: 先验决策图模式
        decision_prior_fusion: 先验图接入方式 ('concat' 或 'mix')
        decision_prior_strength: 先验权重强度
        weight_activation: 权重激活方式 ('softmax' 或 'sigmoid_l1')
    """

    def __init__(self, num_source_images=5, channels=32,
                 use_algae_decision_prior=False,
                 decision_prior_mode='gradient',
                 decision_prior_fusion='concat',
                 decision_prior_strength=0.5,
                 weight_activation='softmax'):
        super().__init__()
        self.num_source_images = num_source_images
        self.use_algae_decision_prior = use_algae_decision_prior
        self.decision_prior_fusion = decision_prior_fusion
        self.decision_prior_strength = decision_prior_strength
        self.weight_activation = weight_activation

        if decision_prior_fusion not in ['concat', 'mix']:
            raise ValueError(f'Unsupported decision prior fusion mode: {decision_prior_fusion}')
        if weight_activation not in ['softmax', 'sigmoid_l1']:
            raise ValueError(f'Unsupported weight activation: {weight_activation}')

        prior_channels = num_source_images if (use_algae_decision_prior and decision_prior_fusion == 'concat') else 0

        # 输入通道：解码器特征 + 每个源图像 (3 通道) + 可选先验决策图
        fusion_channels = channels + num_source_images * 3 + prior_channels

        if use_algae_decision_prior:
            self.decision_prior = AlgaeAwareDecisionPrior(mode=decision_prior_mode)
        else:
            self.decision_prior = None

        # 为每个源图生成权重图
        self.weight_generator = nn.Sequential(
            nn.Conv2d(fusion_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // 2, num_source_images, 1, bias=False)
        )

    def _normalize_weights(self, raw_weights):
        if self.weight_activation == 'softmax':
            return torch.softmax(raw_weights, dim=1)

        weights = torch.sigmoid(raw_weights)
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    def forward(self, features, source_images):
        """
        Args:
            features: 解码器输出特征 (B, C, H, W)
            source_images: 源图像列表，每个 (B, 3, H, W)

        Returns:
            Tensor: 融合图像
        """
        b, c, h, w = features.shape

        # 调整源图像尺寸并拼接
        source_resized = [
            F.interpolate(src, size=(h, w), mode='bilinear', align_corners=False)
            for src in source_images
        ]
        sources_cat = torch.cat(source_resized, dim=1)  # (B, 3*N, H, W)

        prior_weights = None
        fusion_inputs = [features, sources_cat]

        if self.decision_prior is not None:
            prior_weights = self.decision_prior(source_images, target_size=(h, w))
            if self.decision_prior_fusion == 'concat':
                fusion_inputs.append(prior_weights)

        # 拼接解码器特征和源图像/可选先验图
        fusion_input = torch.cat(fusion_inputs, dim=1)

        # 生成学习权重图 (B, N, H, W)
        raw_weights = self.weight_generator(fusion_input)
        learned_weights = self._normalize_weights(raw_weights)

        if prior_weights is not None and self.decision_prior_fusion == 'mix':
            strength = max(0.0, min(float(self.decision_prior_strength), 1.0))
            weights = (1.0 - strength) * learned_weights + strength * prior_weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        else:
            weights = learned_weights

        # 加权融合
        fused = torch.zeros_like(source_resized[0])
        for i, src in enumerate(source_resized):
            fused += weights[:, i:i+1] * src

        return fused


class SimpleFusionHead(nn.Module):
    """
    简单融合头

    直接输出融合图像，不使用决策图。
    """

    def __init__(self, channels=32, out_channels=3):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // 2, out_channels, 1),
            nn.Sigmoid()  # 输出 [0, 1] 范围
        )

    def forward(self, x, source_images):
        """
        简单融合：取所有源图像的平均作为基础，学习残差
        """
        # 源图像平均
        avg = torch.stack(source_images, dim=0).mean(dim=0)

        # 学习残差
        residual = self.head(x)

        return avg + residual


class MSegNet(nn.Module):
    """
    m-SegNet 多聚焦图像融合网络

    结合以下改进：
    1. Depthwise Conv - 轻量化
    2. SPPF - 多尺度特征
    3. BiFPN - 双向特征金字塔
    4. SimAM - 无参注意力

    Args:
        num_source_images: 源图像数量
        in_channels: 输入通道数
        stem_channels: Stem 通道数
        stage_channels: 各阶段通道数
        stage_blocks: 各阶段块数
        use_bifpn: 是否使用 BiFPN
        use_simam: 是否使用 SimAM
        use_fusion_head: 融合头类型 ('decision' 或 'simple')
        multi_source_bifpn_fusion: 多源 stage 特征送入 BiFPN 前的融合方式
        decoder_tail_channels: 解码器最后一层通道数
        decision_weight_activation: 决策图权重归一化方式

    Input:
        source_images: List[Tensor], 长度为 num_source_images 的图像列表
                       每个 Tensor shape: (B, 3, H, W)

    Output:
        Tensor: 融合图像 (B, 3, H, W)
    """

    def __init__(self, num_source_images=5, in_channels=3,
                 stem_channels=32,
                 stage_channels=[32, 64, 128, 256],
                 stage_blocks=[2, 4, 6, 4],
                 use_bifpn=True,
                 use_simam=True,
                 use_fusion_head='decision',
                 use_algae_decision_prior=False,
                 decision_prior_mode='gradient',
                 decision_prior_fusion='concat',
                 decision_prior_strength=0.5,
                 multi_source_bifpn_fusion='first',
                 decoder_tail_channels=16,
                 decision_weight_activation='softmax'):
        super().__init__()

        self.num_source_images = num_source_images
        self.multi_source_bifpn_fusion = multi_source_bifpn_fusion

        if self.multi_source_bifpn_fusion not in ['first', 'mean', 'max']:
            raise ValueError(f'Unsupported multi_source_bifpn_fusion: {self.multi_source_bifpn_fusion}')

        # 共享权重编码器
        self.encoder = Encoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks
        )

        # SPPF 模块
        self.sppf = SPPF(
            in_channels=stage_channels[-1],
            out_channels=stage_channels[-1]
        )

        # BiFPN
        self.use_bifpn = use_bifpn
        if use_bifpn:
            self.bifpn = BiFPN(
                in_channels_list=stage_channels,
                out_channels=128,
                num_levels=4,
                num_layers=2
            )

        # SimAM 注意力
        self.use_simam = use_simam
        if use_simam:
            self.simam = SimAM()

        # 特征融合 (多源图像特征拼接)
        fusion_input_channels = stage_channels[-1] * num_source_images
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_input_channels, 128, 1, bias=False),  # 与 BiFPN 输出通道一致
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True)
        )

        # 解码器
        # 注意：Encoder 有 Stem(2x) + 4 个 Stages(2x×4) = 32 倍下采样
        # 所以 Decoder 需要 5 个 block 来恢复 32 倍
        self.decoder = Decoder(
            encoder_channels=stage_channels[::-1] + [stem_channels],
            decoder_channels=[256, 128, 64, 32, decoder_tail_channels],
            bifpn_channels=128  # BiFPN 输出通道
        )

        fusion_head_channels = decoder_tail_channels

        # 融合头
        if use_fusion_head == 'decision':
            self.fusion_head = DecisionMapFusion(
                num_source_images=num_source_images,
                channels=fusion_head_channels,
                use_algae_decision_prior=use_algae_decision_prior,
                decision_prior_mode=decision_prior_mode,
                decision_prior_fusion=decision_prior_fusion,
                decision_prior_strength=decision_prior_strength,
                weight_activation=decision_weight_activation
            )
        else:
            self.fusion_head = SimpleFusionHead(channels=fusion_head_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _fuse_stage_outputs_for_bifpn(self, encoded_features):
        if self.multi_source_bifpn_fusion == 'first':
            return list(encoded_features[0]['features'][1:])

        stage_outputs = []
        num_stage_levels = len(encoded_features[0]['features']) - 1
        for level in range(num_stage_levels):
            level_feats = [enc['features'][level + 1] for enc in encoded_features]
            stacked = torch.stack(level_feats, dim=0)
            if self.multi_source_bifpn_fusion == 'mean':
                fused_level = stacked.mean(dim=0)
            else:
                fused_level = stacked.max(dim=0)[0]
            stage_outputs.append(fused_level)

        return stage_outputs

    def forward(self, source_images):
        """
        Args:
            source_images: List[Tensor], 源图像列表
                           每个 Tensor shape: (B, 3, H, W)

        Returns:
            Tensor: 融合图像 (B, 3, H, W)
        """
        # 1. 编码每个源图像
        encoded_features = []
        for src in source_images:
            enc_out = self.encoder(src)
            encoded_features.append(enc_out)

        # 2. 为 BiFPN 收集编码器各阶段特征
        # encoder.features[0] = stem 输出，features[1:] = stage1-4 输出
        stage_outputs = self._fuse_stage_outputs_for_bifpn(encoded_features)

        # 3. 对最深层特征应用 SPPF 和 SimAM
        sppf_out = self.sppf(stage_outputs[-1])
        if self.use_simam:
            sppf_out = self.simam(sppf_out)
        stage_outputs[-1] = sppf_out

        # 4. BiFPN 多尺度融合
        if self.use_bifpn:
            bifpn_features = self.bifpn(stage_outputs)  # 4 层特征
        else:
            bifpn_features = stage_outputs

        # 5. 拼接所有源图像的特征
        # 取每个源图像的最深层特征拼接
        concatenated = torch.cat([f['out'] for f in encoded_features], dim=1)
        fused = self.fusion_conv(concatenated)

        # 6. 收集所有源图像的编码器特征用于跳跃连接（5 层：stem + stage1-4）
        avg_encoder_features = []
        for i in range(len(encoded_features[0]['features'])):
            avg_feat = torch.stack([f['features'][i] for f in encoded_features], dim=0).mean(dim=0)
            avg_encoder_features.append(avg_feat)

        decoded = self.decoder(fused, avg_encoder_features, bifpn_features)

        # 7. 融合头生成最终输出
        output = self.fusion_head(decoded, source_images)

        return output


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_source_images=5, **kwargs):
    """
    创建 m-SegNet 模型

    Args:
        num_source_images: 源图像数量
        **kwargs: 其他参数

    Returns:
        MSegNet 模型
    """
    return MSegNet(num_source_images=num_source_images, **kwargs)
