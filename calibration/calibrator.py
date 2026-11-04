"""Camera calibration solver, error evaluator, parameter persistence, and undistortion."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from .board import CheckerboardDetector
from .config import CalibrationConfig


@dataclass
class CameraParameters:
    """Container for calibrated intrinsic camera parameters."""

    camera_matrix: np.ndarray  # 3x3 K matrix
    distortion_coefficients: np.ndarray  # 1x5 or 1x8 D vector
    image_width: int
    image_height: int
    reprojection_error_px: float
    image_count: int
    grid_inner_corners: List[int]
    square_size_mm: float
    timestamp: str

    def to_dict(self) -> dict:
        """Serializes parameters into clean JSON-serializable dictionary."""
        return {
            "metadata": {
                "system": "Measure-It Optical Metrology",
                "calibration_timestamp": self.timestamp,
                "image_count": self.image_count,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "grid_inner_corners": self.grid_inner_corners,
                "square_size_mm": self.square_size_mm,
                "mean_reprojection_error_px": round(float(self.reprojection_error_px), 5),
            },
            "intrinsics": {
                "camera_matrix_K": self.camera_matrix.tolist(),
                "distortion_coefficients_D": self.distortion_coefficients.tolist(),
                "focal_length_x_px": round(float(self.camera_matrix[0, 0]), 3),
                "focal_length_y_px": round(float(self.camera_matrix[1, 1]), 3),
                "principal_point_cx_px": round(float(self.camera_matrix[0, 2]), 3),
                "principal_point_cy_px": round(float(self.camera_matrix[1, 2]), 3),
            },
        }

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Saves camera parameters to JSON file."""
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "CameraParameters":
        """Loads camera parameters from JSON file."""
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Calibration parameter file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        intr = data["intrinsics"]

        return cls(
            camera_matrix=np.array(intr["camera_matrix_K"], dtype=np.float64),
            distortion_coefficients=np.array(intr["distortion_coefficients_D"], dtype=np.float64),
            image_width=int(meta["image_width"]),
            image_height=int(meta["image_height"]),
            reprojection_error_px=float(meta["mean_reprojection_error_px"]),
            image_count=int(meta["image_count"]),
            grid_inner_corners=list(meta["grid_inner_corners"]),
            square_size_mm=float(meta["square_size_mm"]),
            timestamp=str(meta.get("calibration_timestamp", "")),
        )


class CameraCalibrator:
    """Core calibration processor managing image ingestion, optimization, and error auditing."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.detector = CheckerboardDetector(self.config)
        self.object_points: List[np.ndarray] = []
        self.image_points: List[np.ndarray] = []
        self.valid_image_paths: List[Path] = []
        self.failed_image_paths: List[Path] = []
        self.image_shape: Optional[Tuple[int, int]] = None
        self.parameters: Optional[CameraParameters] = None
        self.per_image_errors: List[Dict[str, Union[str, float]]] = []

    def process_image(self, image_path: Union[str, Path]) -> Tuple[bool, Optional[np.ndarray]]:
        """Processes a single calibration image."""
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            self.failed_image_paths.append(path)
            return False, None

        h, w = image.shape[:2]
        if self.image_shape is None:
            self.image_shape = (w, h)
        elif self.image_shape != (w, h):
            raise ValueError(
                f"Image resolution mismatch: {path.name} is {w}x{h}, expected {self.image_shape[0]}x{self.image_shape[1]}"
            )

        found, corners, _ = self.detector.detect_corners(image)
        if found and corners is not None:
            self.object_points.append(self.detector.generate_object_points())
            self.image_points.append(corners)
            self.valid_image_paths.append(path)
            annotated = self.detector.draw_corners(image, corners)
            return True, annotated

        self.failed_image_paths.append(path)
        return False, None

    def process_directory(
        self, directory_path: Union[str, Path], extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    ) -> Dict[str, int]:
        """Scans a directory and processes all calibration images."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Calibration image directory not found: {dir_path}")

        image_files = sorted(
            [p for p in dir_path.iterdir() if p.suffix.lower() in extensions]
        )

        for img_path in image_files:
            self.process_image(img_path)

        return {
            "total_found": len(image_files),
            "valid_detections": len(self.valid_image_paths),
            "failed_detections": len(self.failed_image_paths),
        }

    def calibrate(self) -> CameraParameters:
        """Executes Levenberg-Marquardt optimization to compute camera intrinsics and distortion."""
        valid_count = len(self.valid_image_paths)
        if valid_count < self.config.min_required_images:
            raise ValueError(
                f"Insufficient valid calibration images: found {valid_count}, required at least {self.config.min_required_images}."
            )

        if self.image_shape is None:
            raise RuntimeError("No valid images processed to infer image shape.")

        w, h = self.image_shape

        # Run OpenCV camera calibration with configured flags
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            (w, h),
            None,
            None,
            flags=self.config.calibration_flags,
        )

        # Compute reprojection error per image and global MRE
        total_error = 0.0
        total_points = 0
        self.per_image_errors = []

        for i in range(len(self.object_points)):
            projected_pts, _ = cv2.projectPoints(
                self.object_points[i], rvecs[i], tvecs[i], K, D
            )
            pts_actual = self.image_points[i]
            # Euclidean L2 norm between detected and reprojected corners
            diff = pts_actual.reshape(-1, 2) - projected_pts.reshape(-1, 2)
            sq_dists = np.sum(diff ** 2, axis=1)
            error_sq = float(np.sum(sq_dists))
            n_pts = len(self.object_points[i])
            img_mre = float(np.sqrt(error_sq / n_pts))

            self.per_image_errors.append(
                {
                    "filename": self.valid_image_paths[i].name,
                    "reprojection_error_px": round(img_mre, 4),
                }
            )

            total_error += error_sq
            total_points += n_pts

        global_mre = float(np.sqrt(total_error / total_points))

        self.parameters = CameraParameters(
            camera_matrix=K,
            distortion_coefficients=D,
            image_width=w,
            image_height=h,
            reprojection_error_px=global_mre,
            image_count=valid_count,
            grid_inner_corners=[self.config.grid_cols, self.config.grid_rows],
            square_size_mm=self.config.square_size_mm,
            timestamp=datetime.now().isoformat(),
        )

        return self.parameters

    @staticmethod
    def undistort_image(
        image: np.ndarray,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        alpha: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """Removes radial and tangential lens distortion from an image.

        Args:
            image: Distorted input image.
            camera_matrix: Calibrated 3x3 K matrix.
            distortion_coefficients: Calibrated distortion vector D.
            alpha: Free scaling parameter between 0 (all pixels valid, some cropped) and 1 (all source pixels preserved, some black edges).

        Returns:
            Tuple of:
                - np.ndarray: Undistorted image.
                - np.ndarray: Optimal new camera matrix K_new.
                - Tuple: Valid region of interest (x, y, w, h).
        """
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion_coefficients, (w, h), alpha, (w, h)
        )
        undistorted = cv2.undistort(
            image, camera_matrix, distortion_coefficients, None, new_camera_matrix
        )
        return undistorted, new_camera_matrix, roi
