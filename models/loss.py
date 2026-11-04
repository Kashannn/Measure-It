"""Compound Loss function for sharp contour semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Smooth Dice Loss directly optimizing Intersection-over-Union."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Dice loss from raw model logits."""
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Weighted combination of Binary Cross Entropy and Dice Loss.

    BCE provides smooth gradient updates across all pixels, while Dice loss
    directly penalizes false positives and edge erosion, ensuring crisp
    orthogonal boundaries necessary for physical dimension calculation.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
