"""Geometric contour extraction and metric dimension calculation."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class MetricMeasurementResult:
    """Encapsulates object dimension measurements in pixels and millimetres."""

    found: bool
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    width_px: Optional[float] = None
    height_px: Optional[float] = None
    center_px: Optional[Tuple[float, float]] = None
    rotation_deg: Optional[float] = None
    box_corners: Optional[np.ndarray] = None  # (4, 2) int32 bounding box points
    contour: Optional[np.ndarray] = None
    area_px: Optional[int] = None
    error_message: Optional[str] = None


class MetricContourExtractor:
    """Extracts primary object contour and converts rotated bounding box to millimetres."""

    def __init__(self, min_area_px: int = 5000):
        self.min_area_px = min_area_px

    def extract_dimensions(
        self,
        binary_mask: np.ndarray,
        pixels_per_mm: float,
    ) -> MetricMeasurementResult:
        """Finds primary object contour and turns pixels into millimetres.

        ========================================================================
        CORE FORMULATION (Question K1: 'Which file and line turns pixels into mm?')
        ========================================================================
        The pixel dimensions from cv2.minAreaRect(contour) are divided by the
        spatial scale ratio (PPM = pixels per mm):
            Width_mm  = min(dim1_px, dim2_px) / pixels_per_mm
            Height_mm = max(dim1_px, dim2_px) / pixels_per_mm
        """
        # Find external contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return MetricMeasurementResult(
                found=False,
                error_message="OBJECT_NOT_FOUND: No segmented foreground contour found in mask.",
            )

        # Select largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(largest_contour))

        if area < self.min_area_px:
            return MetricMeasurementResult(
                found=False,
                error_message=f"OBJECT_NOT_FOUND: Largest contour area ({area} px) is below minimum threshold ({self.min_area_px} px).",
            )

        # Compute oriented minimum-area bounding box
        rect = cv2.minAreaRect(largest_contour)
        (cx, cy), (dim1, dim2), angle = rect

        if dim1 <= 0 or dim2 <= 0:
            return MetricMeasurementResult(
                found=False,
                error_message="DEGENERATE_BOUNDING_BOX: Contour produced non-positive dimensions.",
            )

        # Standardize: Width is shorter dimension, Height is longer dimension
        w_px = float(min(dim1, dim2))
        h_px = float(max(dim1, dim2))

        # CONVERT PIXELS TO MILLIMETRES (K1 Answer Anchor)
        width_mm = w_px / pixels_per_mm
        height_mm = h_px / pixels_per_mm

        # 4 box corner coordinates for visualization
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        return MetricMeasurementResult(
            found=True,
            width_mm=round(width_mm, 2),
            height_mm=round(height_mm, 2),
            width_px=round(w_px, 2),
            height_px=round(h_px, 2),
            center_px=(round(cx, 1), round(cy, 1)),
            rotation_deg=round(angle, 2),
            box_corners=box_pts,
            contour=largest_contour,
            area_px=area,
        )
