# Dataset Guide

本项目的数据集采用“图像组”组织方式。每个图像组表示同一显微视野下的 5 张不同焦平面浮游藻类显微图像。

## Directory Layout

推荐目录结构如下：

```text
all_data/
└── split_data/
    ├── train/
    │   ├── group_001/
    │   │   ├── img_1.png
    │   │   ├── img_2.png
    │   │   ├── img_3.png
    │   │   ├── img_4.png
    │   │   └── img_5.png
    │   └── group_002/
    ├── val/
    │   └── group_003/
    └── test/
        └── group_004/
```

代码会遍历数据目录下的子目录，并将每个子目录识别为一个图像组。

## Image Group Rules

- 每个图像组固定包含 5 张源图像
- 5 张图像应来自同一显微视野、不同焦平面
- 支持的常见格式包括 `.jpg`、`.png`、`.bmp`、`.tif`
- 文件会按文件名排序后读取
- 推荐统一命名为 `img_1.png` 到 `img_5.png`
- 本项目当前不使用 `gt.png` 进行训练、监督或参考评估

标准图像组示例：

```text
group_001/
├── img_1.png
├── img_2.png
├── img_3.png
├── img_4.png
└── img_5.png
```

## Resolution

训练和推理可使用 512×512 图像块。对于显微镜采集的高分辨率图像，可以在推理阶段使用瓦片策略切分为多个小图像块，分别融合后再拼接为完整结果。

## Dataset Split

建议将数据划分为：

```text
train/    训练集
val/      验证集
test/     测试集
```

如果数据量较小，可以先使用固定随机种子划分，确保实验可复现。

## Publishing Notice

数据集默认不提交到 GitHub。本仓库的 `.gitignore` 已忽略：

```text
all_data/
data/
datasets/
```

如果需要公开数据，请先确认：

- 数据来源允许再分发
- 图像内容不包含隐私信息
- 数据集许可证允许用于当前开源方式
- README 中包含数据来源和下载方式

## Suggested External Storage

大体积数据推荐放在：

- Hugging Face Datasets
- Zenodo
- Google Drive
- 百度网盘
- 实验室或机构服务器

然后在 README 中提供下载链接、校验信息和使用说明。
