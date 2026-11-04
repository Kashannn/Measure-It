"""Defensive guardrails and input trustworthiness validation for optical metrology.

Directly satisfies Technical Assessment Section 5.4 Item 9:
"Defensive rejection: refusing untrustworthy inputs rather than silently returning bad numbers."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class GuardrailResult:
    """Outcome of input and output trustworthiness checks."""

    passed: bool
    rejection_code: Optional[str] = None  # E.g. "TILT_DETECTED", "POOR_LIGHTING", "BLURRY_IMAGE", "LOW_CONFIDENCE"
    reason: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


class InputGuardrails:
    """Validates operational conditions before allowing measurement numbers to be published."""

    def __init__(
        self,
        max_tilt_skew: float = 1.10,
        min_brightness: float = 40.0,
        max_brightness: float = 240.0,
        min_blur_var: float = 45.0,
        min_model_confidence: float = 0.85,
    ):
        self.max_tilt_skew = max_tilt_skew
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_blur_var = min_blur_var
        self.min_model_confidence = min_model_confidence

    def check_pre_inference(
        self,
        image: np.ndarray,
        tilt_skew_ratio: Optional[float] = None,
    ) -> GuardrailResult:
        """Runs image quality and geometric integrity checks before or during inference."""
        # 1. Illumination Check (mean pixel intensity)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_intensity = float(np.mean(gray))

        if mean_intensity < self.min_brightness:
            return GuardrailResult(
                passed=False,
                rejection_code="POOR_LIGHTING_DARK",
                reason=f"DEFENSIVE REJECTION: Image is severely underexposed (mean brightness {mean_intensity:.1f} < {self.min_brightness}).",
                metric_value=mean_intensity,
                threshold=self.min_brightness,
            )

        if mean_intensity > self.max_brightness:
            return GuardrailResult(
                passed=False,
                rejection_code="POOR_LIGHTING_OVEREXPOSED",
                reason=f"DEFENSIVE REJECTION: Image is overexposed/saturated (mean brightness {mean_intensity:.1f} > {self.max_brightness}).",
                metric_value=mean_intensity,
                threshold=self.max_brightness,
            )

        # 2. Defocus / Motion Blur Check (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.min_blur_var:
            return GuardrailResult(
                passed=False,
                rejection_code="BLURRY_IMAGE",
                reason=f"DEFENSIVE REJECTION: Excessive motion blur or defocus detected (Laplacian variance {laplacian_var:.1f} < {self.min_blur_var}).",
                metric_value=laplacian_var,
                threshold=self.min_blur_var,
            )

        # 3. Camera Tilt / Perspective Skew Check
        if tilt_skew_ratio is not None and tilt_skew_ratio > self.max_tilt_skew:
            return GuardrailResult(
                passed=False,
                rejection_code="EXCESSIVE_CAMERA_TILT",
                reason=f"DEFENSIVE REJECTION: Severe out-of-plane camera tilt detected (perspective skew ratio {tilt_skew_ratio:.3f} > {self.max_tilt_skew}). Coplanar planar assumption broken.",
                metric_value=tilt_skew_ratio,
                threshold=self.max_tilt_skew,
            )

        return GuardrailResult(passed=True)

    def check_post_inference(
        self,
        confidence: float,
        width_mm: float,
        height_mm: float,
    ) -> GuardrailResult:
        """Validates neural segmentation confidence and physical plausibility."""
        # 1. Neural Segmentation Confidence Check
        if confidence < self.min_model_confidence:
            return GuardrailResult(
                passed=False,
                rejection_code="LOW_CONFIDENCE",
                reason=f"DEFENSIVE REJECTION: Segmentation confidence ({confidence:.3f}) is below minimum threshold ({self.min_model_confidence}). Mask boundary untrustworthy.",
                metric_value=confidence,
                threshold=self.min_model_confidence,
            )

        # 2. Aspect Ratio Anomaly Check (nominal aspect ratio is ~1.58)
        aspect = height_mm / max(width_mm, 1e-4)
        if aspect < 0.90 or aspect > 2.80:
            return GuardrailResult(
                passed=False,
                rejection_code="ANOMALOUS_ASPECT_RATIO",
                reason=f"DEFENSIVE REJECTION: Segmented bounding box aspect ratio ({aspect:.2f}) deviates from expected physical envelope [0.90, 2.80].",
                metric_value=aspect,
                threshold=2.80,
            )

        return GuardrailResult(passed=True)
