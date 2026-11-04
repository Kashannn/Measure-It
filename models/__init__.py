"""Neural semantic segmentation package for industrial optical metrology.

Implements custom DeepLabV3+ architecture with ResNet-34 encoder,
compound BCE + Dice loss, training routines, and COCO/Pascal VOC evaluation metrics.
"""

from .model import build_model, MetrologySegmentationModel
from .loss import BCEDiceLoss
from .metrics import evaluate_segmentation_metrics

__all__ = [
    "build_model",
    "MetrologySegmentationModel",
    "BCEDiceLoss",
    "evaluate_segmentation_metrics",
]
