"""Checkerboard corner detection and sub-pixel refinement module."""

from typing import Optional, Tuple
import cv2
import numpy as np
from .config import CalibrationConfig


class CheckerboardDetector:
    """Robust checkerboard corner detector with sub-pixel refinement."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.grid_size = self.config.grid_size
        self.square_size_mm = self.config.square_size_mm

    def generate_object_points(self) -> np.ndarray:
        """Generates 3D coordinates of inner corners in physical world frame (mm).

        Assumes the planar board lies on Z = 0: (0,0,0), (25,0,0), (50,0,0)...
        Returns:
            np.ndarray: Float32 array of shape (N, 3), where N = cols * rows.
        """
        cols, rows = self.grid_size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size_mm
        return objp

    def detect_corners(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects checkerboard inner corners with sub-pixel optimization.

        Args:
            image: BGR or Grayscale input image.

        Returns:
            Tuple of:
                - bool: True if all inner corners were detected successfully.
                - np.ndarray or None: Sub-pixel refined 2D corners of shape (N, 1, 2).
                - np.ndarray or None: Grayscale image used for detection.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )

        found, corners = cv2.findChessboardCorners(gray, self.grid_size, flags)

        if not found or corners is None:
            return False, None, gray

        # Sub-pixel refinement
        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=self.config.subpix_win_size,
            zeroZone=self.config.subpix_zero_zone,
            criteria=self.config.subpix_criteria,
        )

        return True, refined_corners, gray

    def draw_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Overlays detected corner markers and index connectivity lines on image."""
        annotated = image.copy()
        cv2.drawChessboardCorners(annotated, self.grid_size, corners, True)
        return annotated
