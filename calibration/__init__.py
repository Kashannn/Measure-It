"""Camera Calibration Package for Industrial Optical Metrology.

Implements OpenCV-based camera calibration, sub-pixel corner refinement,
pinhole intrinsic/extrinsic parameter estimation, and lens undistortion.
"""

from .config import CalibrationConfig
from .board import CheckerboardDetector
from .calibrator import CameraCalibrator

__all__ = ["CalibrationConfig", "CheckerboardDetector", "CameraCalibrator"]
