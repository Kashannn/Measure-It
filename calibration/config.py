"""Configuration and parameters for camera calibration."""

from dataclasses import dataclass, asdict
from typing import Tuple
import cv2


@dataclass(frozen=True)
class CalibrationConfig:
    """Rigid configuration parameters for checkerboard camera calibration."""

    # Grid inner corners (number of internal vertices, not squares)
    grid_cols: int = 9
    grid_rows: int = 7

    # Physical dimensions of each square in millimetres
    square_size_mm: float = 25.0

    # Sub-pixel corner refinement search window half-size
    subpix_win_size: Tuple[int, int] = (11, 11)
    subpix_zero_zone: Tuple[int, int] = (-1, -1)

    # Sub-pixel termination criteria: 30 iterations or eps < 0.001 px
    max_subpix_iters: int = 30
    subpix_eps: float = 0.001

    # Calibration solver flags (0 for standard 5-param Brown-Conrady: k1, k2, p1, p2, k3)
    calibration_flags: int = 0

    # Maximum acceptable Mean Reprojection Error in pixels
    max_acceptable_mre_px: float = 0.50

    # Minimum required valid calibration image count (Assessment mandate: >= 20)
    min_required_images: int = 20

    @property
    def grid_size(self) -> Tuple[int, int]:
        """Returns (cols, rows) tuple representing inner corners."""
        return self.grid_cols, self.grid_rows

    @property
    def subpix_criteria(self) -> Tuple[int, int, float]:
        """Returns OpenCV termination criteria tuple."""
        return (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.max_subpix_iters,
            self.subpix_eps,
        )

    def to_dict(self) -> dict:
        """Converts config to dictionary representation."""
        d = asdict(self)
        d["grid_size"] = list(self.grid_size)
        return d
