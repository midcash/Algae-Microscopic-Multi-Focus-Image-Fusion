"""
SimAM 无参注意力模块

Simple Attention Module - 基于能量函数优化，
无需额外参数即可推断 3D 注意力权重。

参考论文：SimAM: A Simple, Parameter-Free Attention Module for
Convolutional Neural Networks (ICML 2021)
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    SimAM 无参注意力模块

    基于神经元空间重要性理论，通过能量函数优化推断注意力权重，
    无需学习任何额外参数。

    原理通俗解释：
    想象你在看一张特征图，上面每个像素点都是一个“神经元”。
    如果某个神经元的值和周围所有神经元明显不同（“鹤立鸡群”），
    那它很可能很重要，应该给它更高的关注度。
    SimAM 就是通过计算每个神经元的“与众不同程度”，
    自动生成一个注意力权重，不需要额外训练参数。

    Args:
        e_lambda: 正则化参数，防止除零 (默认 1e-4)

    Shape:
        Input: (N, C, H, W)  — N: 批量大小, C: 通道数, H: 高, W: 宽
        Output: (N, C, H, W) —— 尺寸不变，但每个位置的值被注意力权重缩放

    Equation:
        1. 计算每个神经元的能量：
           e_t = (x_t - μ)² / (1/n * Σ(x_i - μ)² + λ)

        其中:
           - x_t: 目标神经元
           - μ: 通道均值
           - n: 像素数量
           - λ: 正则化参数

        2. 应用 sigmoid 得到注意力权重
        3. 原始特征 × 注意力权重
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda   # 一个很小的常数，防止方差为零时除零报错

    def forward(self, x):
        # x 的形状： (b, c, h, w)
        b, c, h, w = x.size()
        # n 是用来计算方差的归一化系数（论文中的 n-1，即除了当前神经元本身）
        n = w * h - 1  # 减去中心像素

        # ---------- 第 1 步：求每个通道的平均值 ----------
        # 对于每个通道，把该通道内所有空间位置（h*w 个像素）的值求平均
        # keepdim=True 保持维度，方便后面相减时进行广播
        x_mean = x.mean(dim=[2, 3], keepdim=True)

        # ---------- 第 2 步：计算 (x - μ)² ----------
        # 这个量衡量“当前神经元的值偏离通道平均值的程度”
        # 偏离越大，意味着这个神经元越不寻常
        x_minus_mu_square = (x - x_mean) ** 2

        # ---------- 第 3 步：计算像素总体的方差估计 ----------
        # 将所有空间位置的偏离平方求和，然后除以 n（像素总数-1）
        # 得到的是整个通道的“平均偏离程度”，也就是方差
        variance = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n

        # ---------- 第 4 步：计算能量值 ----------
        # 能量 = (当前神经元偏离平方) / (4 * (方差 + 极小常数))
        # 能量越高 → 当前神经元相对于整个通道的背景越突出 → 越重要
        # 分母的 4 是论文中推导出来的常数，不影响相对大小
        energy = x_minus_mu_square / (4 * (variance + self.e_lambda))

        # ---------- 第 5 步：Sigmoid 转为注意力权重 ----------
        # sigmoid 把任意值压缩到 0～1 之间，作为权重因子
        # 能量越大 → sigmoid 输出越接近 1 → 该位置被加强
        # 能量越小 → sigmoid 输出越接近 0.5 → 该位置被适当保留
        attention = torch.sigmoid(energy)

        # ---------- 第 6 步：用注意力加权原特征 ----------
        # 逐元素相乘，重要位置的数值被放大，不重要的被抑制
        return x * attention

    def extra_repr(self) -> str:
        return f'e_lambda={self.e_lambda}'


class SimAMWithGate(nn.Module):
    """
    SimAM 注意力模块 (带门控机制)

    在基础 SimAM 基础上添加可学习的门控参数，
    可以控制注意力强度的自适应调整。

    Args:
        e_lambda: 正则化参数
        gate_init: 门控初始值 (0=关闭，1=全开)
    """

    def __init__(self, e_lambda=1e-4, gate_init=0.5):
        super().__init__()
        self.e_lambda = e_lambda
        # 将门控参数定义为可学习的 Parameter，初始为 gate_init
        self.gate = nn.Parameter(torch.tensor(gate_init))
        # torch.tensor(gate_init) —— 创建一个普通的 0 维张量（标量），数值为 gate_init（默认 0.5）。
        # nn.Parameter(...) —— 把这个普通张量包装成一个“可学习的参数”，告诉 PyTorch：“这个数字是要通过训练调整的，请帮我跟踪它！”
        # self.gate = ... —— 把这个参数绑定为模块的一个属性。

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        # --- 以下同基础 SimAM，计算注意力权重 ---
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - x_mean) ** 2
        variance = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        energy = x_minus_mu_square / (4 * (variance + self.e_lambda))
        attention = torch.sigmoid(energy)  # sigmoid 函数的公式是：σ(x) = 1 / (1 + e⁻ˣ)   torch.sigmoid(x)：把任意实数 x 映射到 (0,1) 区间。

        # --- 门控机制 ---
        # gate 是一个标量参数，经过 sigmoid 后限制在 0~1 之间
        gate_factor = torch.sigmoid(self.gate)

        # 注意力应用方式：原始特征 + gate_factor * attention * 原始特征
        # 等价于 x * (1 + gate_factor * attention)
        # 当 gate_factor=0 时，退化为原始特征（不加注意力）
        # 当 gate_factor=1 时，相当于特征被放大 (1 + attention) 倍
        # 因为 attention 在 0~1，所以增强因子在 1~2 之间
        return x * (1 + gate_factor * attention)

    def extra_repr(self) -> str:
        return f'e_lambda={self.e_lambda}, gate_init=0.5'
