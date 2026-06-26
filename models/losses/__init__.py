"""
损失函数模块
"""

from models.losses.fusion_loss import FusionLoss
from models.losses.boundary_loss import BoundaryLoss, sobel_gradient
from models.losses.multi_scale_loss import MultiScaleLoss
from models.losses.morphology_loss import MorphologyPreservingLoss

__all__ = [
    'FusionLoss',
    'BoundaryLoss',
    'MultiScaleLoss',
    'MorphologyPreservingLoss',
    'sobel_gradient'
]
