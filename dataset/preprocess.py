"""Preprocessing module: Applies camera matrix undistortion to image captures."""

from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

from calibration.calibrator import CameraParameters


class ImagePreprocessor:
    """Handles optical undistortion and standardization of dataset images."""

    def __init__(self, camera_params_path: Union[str, Path] = "calibration/camera_params.json"):
        self.params_path = Path(camera_params_path)
        if not self.params_path.exists():
            raise FileNotFoundError(
                f"Camera parameters not found at {self.params_path}. Run calibration first!"
            )
        self.params = CameraParameters.load_json(self.params_path)
        self.K = self.params.camera_matrix
        self.D = self.params.distortion_coefficients

    def undistort(self, image: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """Undistorts an input image using the calibrated K and D parameters.

        Args:
            image: Distorted BGR image.
            alpha: 0.0 crops invalid edge pixels, 1.0 retains all pixels with black padding.

        Returns:
            np.ndarray: Undistorted rectilinear image.
        """
        h, w = image.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), alpha, (w, h))
        undistorted = cv2.undistort(image, self.K, self.D, None, new_K)
        return undistorted

    def process_file(
        self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Loads a raw photo, removes lens distortion, and writes to disk."""
        in_p = Path(input_path)
        out_p = Path(output_path) if output_path else in_p

        image = cv2.imread(str(in_p))
        if image is None:
            raise ValueError(f"Could not read image: {in_p}")

        rectified = self.undistort(image)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), rectified, [cv2.IMWRITE_JPEG_QUALITY, 96])
        return out_p
