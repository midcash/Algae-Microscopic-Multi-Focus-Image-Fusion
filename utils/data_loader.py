"""
数据加载器

支持多聚焦图像数据集的加载和预处理。
"""

import os
import glob
import random

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


class MultiFocusDataset(Dataset):
    """
    多聚焦图像数据集

    期望目录结构:
        all_data/
        └── train/
            ├── group_001/
            │   ├── img_1.jpg
            │   ├── img_2.jpg
            │   ├── img_3.jpg
            │   ├── img_4.jpg
            │   └── img_5.jpg
            ├── group_002/
            └── ...

    或 (带 Ground Truth):
        all_data/
        └── train/
            ├── group_001/
            │   ├── img_1.jpg
            │   ├── img_2.jpg
            │   ├── ...
            │   └── gt.jpg  # Ground Truth
            └── ...

    Args:
        root_dir: 数据集根目录
        is_train: 是否训练模式 (启用增强)
        input_size: 输入图像尺寸
        num_source_images: 源图像数量
        augment: 是否数据增强
    """

    def __init__(self, root_dir, is_train=True, input_size=512,
                 num_source_images=5, augment=True):
        super().__init__()

        self.root_dir = root_dir
        self.is_train = is_train
        self.input_size = input_size
        self.num_source_images = num_source_images
        self.augment = augment and is_train

        # 查找所有图像组
        self.groups = self._find_groups()
        print(f'Found {len(self.groups)} image groups in {root_dir}')

    def _find_groups(self):
        """查找所有图像组"""
        groups = []

        # 查找所有子目录
        for subdir in sorted(os.listdir(self.root_dir)):
            subpath = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subpath):
                continue

            # 查找所有图像
            images = []
            gt_path = None

            for ext in ['*.jpg', '*.png', '*.bmp', '*.tif']:
                for img_path in glob.glob(os.path.join(subpath, ext)):
                    if 'gt' in os.path.basename(img_path).lower():
                        gt_path = img_path
                    else:
                        images.append(img_path)

            if len(images) >= self.num_source_images:
                # 排序以确保一致性
                images = sorted(images)[:self.num_source_images]
                groups.append({
                    'images': images,
                    'gt': gt_path
                })

        return groups

    def _load_image(self, path):
        """加载单张图像"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f'Failed to load: {path}')

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整大小
        if self.input_size > 0:
            img = cv2.resize(img, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)

        return img

    def _augment_per_focal_plane(self, images, num_planes=None):
        """
        基于空间频率掩码的焦平面增强 — 模拟真实显微景深效果。

        核心思想：每个焦平面图像本身是清晰成像，但由于景深限制，
        图像中只有部分区域在焦、部分区域离焦。离焦区域呈现模糊退化。

        实现方式：
        1. 为每张图生成随机的"清晰-模糊"空间掩码（不同图互补）
        2. 清晰区域保持原样，模糊区域做高斯退化
        3. 综合起来：5张图各有一部分清晰区域，拼接起来 = 全图清晰
        4. 额外：噪声独立添加，模拟传感器噪声

        R_i: 新增源图顺序随机置换，打破固定焦距顺序模式

        Args:
            images: 输入图像列表 [H, W, 3] uint8
            num_planes: 焦平面数量，默认 len(images)
        Returns:
            退化后的图像列表
        """
        if num_planes is None:
            num_planes = len(images)

        h, w = images[0].shape[:2]

        # R_i: 源图顺序随机置换（强制模型学习"哪个最清晰"而非记忆位置）
        if random.random() > 0.3:  # 70% 概率置换
            perm = list(range(num_planes))
            random.shuffle(perm)
            images = [images[p] for p in perm]

        # R_i: Random Crop —— 从 h×w 随机裁剪 0.5-1.0 倍区域
        crop_ratio = random.uniform(0.5, 1.0)
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        if crop_h < 64 or crop_w < 64:
            crop_h, crop_w = 64, 64
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        # 执行裁剪 + 缩放回原尺寸
        if crop_ratio < 0.95:
            images = [img[top:top+crop_h, left:left+crop_w] for img in images]
            images = [cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) for img in images]

        # 1️⃣ 随机几何增强 (所有图一致)
        if random.random() > 0.5:
            images = [cv2.flip(img, 1) for img in images]

        if random.random() > 0.5:
            images = [cv2.flip(img, 0) for img in images]

        angle = random.uniform(-30, 30)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        images = [cv2.warpAffine(img, M, (w, h)) for img in images]

        # 2️⃣ 色彩抖动 (所有图一致，模拟光照变化)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-10, 10)
            images = [cv2.convertScaleAbs(img, alpha=alpha, beta=beta) for img in images]

        # ===== 3️⃣ 生成空间频率掩码 =====
        # 将图像划分为 NxN 块，每块分配给一个焦平面作为"清晰区域"
        # 其余焦平面在该块区域是模糊的
        grid_size = random.choice([6, 8, 10])  # 块粒度：6x6/8x8/10x10 (必须 >= num_planes)
        block_h, block_w = h // grid_size, w // grid_size

        # 为每个块分配一个"主清晰"焦平面
        # 使用 Perlin-like 平滑分配，使清晰区域形成连续的自然斑块
        labels = np.zeros((grid_size, grid_size), dtype=int)
        # 从中心种子开始扩散，生成自然过渡的清晰区域分布
        centers_x = np.random.choice(grid_size, size=num_planes, replace=False)
        centers_y = np.random.choice(grid_size, size=num_planes, replace=False)
        for gy in range(grid_size):
            for gx in range(grid_size):
                # 分配到最近的种子点
                dists = [abs(gx - cx) + abs(gy - cy) for cx, cy in zip(centers_x, centers_y)]
                labels[gy, gx] = np.argmin(dists)

        # 为每个焦平面生成它的"模糊区域"掩码（= 不属于它的块）
        # 并做边缘羽化，避免块间硬边界
        plane_masks = []
        for p in range(num_planes):
            # 1 = 该平面清晰, 0 = 该平面模糊
            block_mask = (labels == p).astype(np.float32)
            # 放大到图像尺寸
            mask = cv2.resize(block_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            # 高斯羽化边缘，使过渡自然
            mask = cv2.GaussianBlur(mask, (block_w * 2 + 1 | 1, block_h * 2 + 1 | 1), 0)
            plane_masks.append(mask)  # [0, 1] float

        # ===== 4️⃣ 按空间掩码退化每张图 =====
        for i in range(num_planes):
            # 该平面应该模糊的区域 = 1 - plane_masks[i]
            blur_mask = 1.0 - plane_masks[i]

            # --- 根据掩码做空间退化 ---
            # 模糊强度根据距离主区域的远近分级
            blur_kernel_sizes = [3, 5, 7, 9]
            blurred_variants = {}
            for ks in blur_kernel_sizes:
                blurred_variants[ks] = cv2.GaussianBlur(images[i], (ks, ks), 0)

            # 构造多级模糊结果：不同区域用不同模糊强度
            # 我们简单做两级：轻度模糊(ks=3)和重度模糊(ks=9)
            mild_blur = cv2.GaussianBlur(images[i], (3, 3), 0)
            heavy_blur = cv2.GaussianBlur(images[i], (9, 9), 0)

            # 用 blur_mask 做空间混合：模糊区域用 heavy_blur，清晰区域用原图
            # 中间过渡区用 mild_blur 混合
            transition = cv2.GaussianBlur(blur_mask, (15, 15), 0)  # 柔化过渡
            # 三路混合
            result = (
                images[i].astype(np.float32) * (1 - transition[..., None]) +
                mild_blur.astype(np.float32) * (transition[..., None] * 0.5) +
                heavy_blur.astype(np.float32) * (transition[..., None] * 0.5)
            )
            images[i] = np.clip(result, 0, 255).astype(np.uint8)

            # --- 每张图独立高斯噪声 ---
            if random.random() < 0.6:
                sigma = random.uniform(0.003, 0.015)
                noise = np.random.normal(0, sigma, (h, w, 3)).astype(np.float32) * 255.0
                images[i] = np.clip(images[i].astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return images

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        # 加载图像
        images = [self._load_image(p) for p in group['images']]

        # 数据增强 (R5: 焦平面独立退化)
        if self.augment:
            images = self._augment_per_focal_plane(images, self.num_source_images)

        # 加载 Ground Truth (如果有)
        gt = None
        if group['gt'] and os.path.exists(group['gt']):
            gt = self._load_image(group['gt'])

        # 归一化到 [0, 1] 并转换为 Tensor
        images_tensor = [
            torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            for img in images
        ]

        result = {
            'sources': images_tensor,
            'paths': group['images']
        }

        if gt is not None:
            result['target'] = torch.from_numpy(
                gt.astype(np.float32) / 255.0
            ).permute(2, 0, 1)
        else:
            # 无 ground truth 时，使用第一张源图像作为占位 target
            # 训练代码中的损失函数不需要 target
            result['target'] = images_tensor[0]

        return result


def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4,
                      pin_memory=True, drop_last=False):
    """
    创建 DataLoade

    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作线程数
        pin_memory: 是否锁定内存
        drop_last: 是否丢弃最后不完整的 batch

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )


class SyntheticDataset(Dataset):
    """
    合成多聚焦数据集

    从清晰图像生成合成多聚焦图像 (用于无 Ground Truth 场景)。

    Args:
        clear_images_dir: 清晰图像目录
        num_focus_planes: 焦平面数量
        blur_sigmas: 模糊参数列表
    """

    def __init__(self, clear_images_dir, num_focus_planes=5,
                 blur_sigmas=None):
        super().__init__()

        if blur_sigmas is None:
            blur_sigmas = [0, 1, 2, 4, 8]  # 默认模糊参数

        self.num_focus_planes = num_focus_planes
        self.blur_sigmas = blur_sigmas

        # 加载清晰图像
        self.clear_images = []
        for ext in ['*.jpg', '*.png', '*.bmp']:
            self.clear_images.extend(glob.glob(
                os.path.join(clear_images_dir, ext)
            ))

        print(f'Found {len(self.clear_images)} clear images')

    def _generate_focus_stack(self, clear_img):
        """生成多聚焦图像栈"""
        images = []

        for i, sigma in enumerate(self.blur_sigmas):
            if sigma == 0:
                # 清晰图像 (添加一些随机噪声模拟真实场景)
                img = clear_img.copy()
                noise = np.random.normal(0, 2, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
            else:
                # 高斯模糊
                ksize = int(sigma * 4 + 1) | 1  # 确保是奇数
                img = cv2.GaussianBlur(clear_img, (ksize, ksize), sigma)

            images.append(img)

        return images

    def __len__(self):
        return len(self.clear_images)

    def __getitem__(self, idx):
        # 加载清晰图像
        clear_path = self.clear_images[idx]
        clear_img = cv2.imread(clear_path)
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.resize(clear_img, (512, 512))

        # 生成多聚焦栈
        focus_stack = self._generate_focus_stack(clear_img)

        # 转换为 Tensor
        images_tensor = [
            torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            for img in focus_stack
        ]

        # 清晰图像作为 Ground Truth
        gt_tensor = torch.from_numpy(
            clear_img.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        return {
            'sources': images_tensor,
            'target': gt_tensor,
            'paths': [clear_path] * len(images_tensor)
        }
