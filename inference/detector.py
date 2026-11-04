"""Reference fiducial detector and metric scale calculator."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class ReferenceDetectionResult:
    """Encapsulates the result of reference fiducial detection."""

    found: bool
    corners: Optional[np.ndarray] = None  # (4, 2) float32 coordinates
    marker_id: Optional[int] = None
    pixels_per_mm: Optional[float] = None
    mean_edge_px: Optional[float] = None
    tilt_skew_ratio: Optional[float] = None
    error_message: Optional[str] = None


class ReferenceDetector:
    """Detects metric ArUco fiducials and computes spatial scale factor (PPM)."""

    def __init__(
        self,
        marker_size_mm: float = 50.0,
        dict_name: int = cv2.aruco.DICT_4X4_50,
        target_marker_id: int = 0,
    ):
        self.marker_size_mm = marker_size_mm
        self.target_marker_id = target_marker_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)

        # Compatible with modern OpenCV (4.7+) and legacy versions
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        else:
            self.detector = None

    def detect(self, image: np.ndarray) -> ReferenceDetectionResult:
        """Detects the reference marker and calculates pixels-per-millimeter.

        Args:
            image: BGR or Grayscale image.

        Returns:
            ReferenceDetectionResult with calculated scale or descriptive failure.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if self.detector is not None:
            corners_list, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners_list, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if ids is None or len(ids) == 0:
            return ReferenceDetectionResult(
                found=False,
                error_message="REFERENCE_NOT_FOUND: No ArUco fiducial detected in image.",
            )

        # Find target marker ID
        ids_flat = ids.flatten()
        target_idx = None
        for i, mid in enumerate(ids_flat):
            if mid == self.target_marker_id:
                target_idx = i
                break

        if target_idx is None:
            # Fall back to first detected marker if specific ID is missing
            target_idx = 0

        corners = corners_list[target_idx].reshape(4, 2).astype(np.float32)
        marker_id = int(ids_flat[target_idx])

        # Compute Euclidean lengths of all 4 edges:
        # e0: (p0->p1), e1: (p1->p2), e2: (p2->p3), e3: (p3->p0)
        edges = [
            float(np.linalg.norm(corners[0] - corners[1])),
            float(np.linalg.norm(corners[1] - corners[2])),
            float(np.linalg.norm(corners[2] - corners[3])),
            float(np.linalg.norm(corners[3] - corners[0])),
        ]

        mean_edge_px = float(np.mean(edges))
        if mean_edge_px < 10.0:
            return ReferenceDetectionResult(
                found=False,
                error_message="REFERENCE_TOO_SMALL: Marker size in pixels is below reliable resolution.",
            )

        # Compute spatial scale: Pixels per Millimetre
        # CRITICAL METROLOGY FORMULATION:
        # PPM = Edge Length (pixels) / Known Physical Dimension (mm)
        pixels_per_mm = mean_edge_px / self.marker_size_mm

        # Tilt skew ratio: ratio of max edge to min edge
        skew_ratio = float(max(edges) / min(edges))

        return ReferenceDetectionResult(
            found=True,
            corners=corners,
            marker_id=marker_id,
            pixels_per_mm=pixels_per_mm,
            mean_edge_px=mean_edge_px,
            tilt_skew_ratio=skew_ratio,
        )
