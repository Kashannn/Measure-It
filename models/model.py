"""Neural segmentation model architecture for physical object outline extraction."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml


class MetrologySegmentationModel(nn.Module):
    """DeepLabV3+ Semantic Segmentation Network with ResNet-34 Encoder.

    ========================================================================
    ARCHITECTURAL JUSTIFICATION (Requirement 3: 'Say why you chose yours')
    ========================================================================
    1. STRICT REGULATORY COMPLIANCE:
       - Ultralytics YOLO and Roboflow models are strictly forbidden by Section 4.2.
       - DeepLabV3+ implemented via SMP provides an open, standard, inspectable PyTorch
         network free of proprietary licensing or cloud dependencies.

    2. SPATIAL PYRAMID DILATED CONVOLUTIONS:
       - Industrial metrology relies on exact boundary detection. Standard CNN decoders
         frequently erode orthogonal edges due to successive strided downsampling.
       - DeepLabV3+ utilizes Atrous Spatial Pyramid Pooling (ASPP) with multi-rate
         dilations (rates: 6, 12, 18), allowing the model to capture multi-scale context
         and retain high-frequency edge gradients simultaneously.

    3. RESNET-34 ENCODER EFFICIENCY:
       - ResNet-34 strikes an optimal balance between expressive feature representation
         and low parameter count (~26.7M parameters).
       - Converges rapidly on small custom datasets (<80 samples) with ImageNet initialization,
         avoiding the severe overfitting risks of ResNet-101 while outperforming lightweight
         MobileNets on sub-millimeter contour precision.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.classes = classes

        self.net = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # Return raw logits for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLabV3+ returning raw logits."""
        return self.net(x)

    @torch.no_grad()
    def predict_mask(
        self,
        image_tensor: torch.Tensor,
        threshold: float = 0.50,
    ) -> Tuple[np.ndarray, float]:
        """Inference helper converting input tensor into clean binary mask and confidence.

        Args:
            image_tensor: Float tensor of shape (1, 3, H, W) normalized to ImageNet stats.
            threshold: Sigmoid probability cutoff (default: 0.50).

        Returns:
            Tuple of:
                - np.ndarray: Uint8 binary mask (0 or 255) of shape (H, W).
                - float: Mean confidence score of foreground predictions.
        """
        self.eval()
        logits = self.forward(image_tensor)
        probs = torch.sigmoid(logits)

        # Foreground probability map
        probs_np = probs.squeeze().cpu().numpy()
        binary_mask = (probs_np >= threshold).astype(np.uint8) * 255

        # Confidence: average certainty in predicted foreground region
        fg_mask = binary_mask > 0
        if np.any(fg_mask):
            confidence = float(np.mean(probs_np[fg_mask]))
        else:
            confidence = 0.0

        return binary_mask, confidence


def build_model(config: Optional[Union[Dict, str, Path]] = None) -> MetrologySegmentationModel:
    """Factory creating DeepLabV3+ model from config dict or YAML path."""
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = {}

    encoder = cfg.get("encoder_name", "resnet34")
    weights = cfg.get("encoder_weights", "imagenet")
    in_channels = cfg.get("in_channels", 3)
    classes = cfg.get("classes", 1)

    return MetrologySegmentationModel(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
    )
