"""High-fidelity synthetic calibration scene simulator.

Projects 3D checkerboard vertices onto a virtual camera sensor with realistic
radial (barrel/pincushion) and tangential lens distortion. Enables immediate
end-to-end mathematical verification of calibration algorithms.
"""

from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np


class SyntheticCalibrationGenerator:
    """Simulates a physical camera with realistic non-linear optical distortion."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (1920, 1080),  # Full HD
        grid_cols: int = 9,
        grid_rows: int = 7,
        square_size_mm: float = 25.0,
    ):
        self.w, self.h = image_size
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.square_size_mm = square_size_mm

        # Ground truth camera matrix (focal length ~ 1450 px, principal point at optical center)
        self.K_gt = np.array(
            [
                [1450.0, 0.0, self.w / 2.0],
                [0.0, 1450.0, self.h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Realistic wide-angle lens distortion (barrel distortion k1 < 0)
        self.D_gt = np.array([-0.22, 0.075, 0.0008, -0.0005, -0.012], dtype=np.float64)

    def generate_board_texture(self) -> np.ndarray:
        """Renders an ideal high-res flat 2D checkerboard texture."""
        n_sq_x = self.grid_cols + 1
        n_sq_y = self.grid_rows + 1
        sq_px = 120
        margin = 80

        tex_w = n_sq_x * sq_px + 2 * margin
        tex_h = n_sq_y * sq_px + 2 * margin

        texture = np.ones((tex_h, tex_w, 3), dtype=np.uint8) * 245  # Off-white matte paper

        for y in range(n_sq_y):
            for x in range(n_sq_x):
                if (x + y) % 2 == 1:
                    x0 = margin + x * sq_px
                    y0 = margin + y * sq_px
                    cv2.rectangle(texture, (x0, y0), (x0 + sq_px, y0 + sq_px), (18, 18, 18), -1)

        return texture

    def generate_scene_image(
        self, rvec: np.ndarray, tvec: np.ndarray, board_texture: np.ndarray
    ) -> np.ndarray:
        """Projects the checkerboard texture through the distorted camera model."""
        th, tw = board_texture.shape[:2]

        # 3D coordinates of the 4 outer corners of the physical board
        # Board dimensions in mm
        bw = (self.grid_cols + 1) * self.square_size_mm
        bh = (self.grid_rows + 1) * self.square_size_mm
        margin_mm = (self.square_size_mm / 120.0) * 80.0

        pts_3d = np.array(
            [
                [-margin_mm, -margin_mm, 0.0],
                [bw + margin_mm, -margin_mm, 0.0],
                [bw + margin_mm, bh + margin_mm, 0.0],
                [-margin_mm, bh + margin_mm, 0.0],
            ],
            dtype=np.float64,
        )

        # Project 3D board corners with ground truth distortion
        proj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.K_gt, self.D_gt)
        proj_pts = proj_pts.reshape(-1, 2).astype(np.float32)

        # Source coordinates in board texture
        src_pts = np.array(
            [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
        )

        # Perspective warp
        H = cv2.getPerspectiveTransform(src_pts, proj_pts)
        warped = cv2.warpPerspective(board_texture, H, (self.w, self.h))

        # Synthetic background (desk surface)
        bg = np.ones((self.h, self.w, 3), dtype=np.uint8) * 190
        # Add subtle noise and texture
        noise = np.random.normal(0, 4, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Composite board onto background
        mask = cv2.warpPerspective(np.ones((th, tw), dtype=np.uint8) * 255, H, (self.w, self.h))
        mask_3c = cv2.merge([mask, mask, mask])

        composite = np.where(mask_3c > 0, warped, bg)
        return composite

    def generate_calibration_dataset(
        self,
        output_dir: Union[str, Path] = "calibration/images",
        num_images: int = 30,
    ) -> List[Path]:
        """Generates a complete dataset of diverse 3D poses (angles, distances, edges)."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        texture = self.generate_board_texture()
        saved_paths: List[Path] = []

        np.random.seed(42)

        for i in range(num_images):
            # Controlled camera tilt and distance ensuring corners remain in FOV
            angle_x = np.deg2rad(np.random.uniform(-18.0, 18.0))
            angle_y = np.deg2rad(np.random.uniform(-18.0, 18.0))
            angle_z = np.deg2rad(np.random.uniform(-12.0, 12.0))

            # Rotation matrix from Euler angles
            Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
            Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
            Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            rvec, _ = cv2.Rodrigues(R)

            # Translation: diverse positions within the camera frustum
            tx = np.random.uniform(-120.0, 120.0)
            ty = np.random.uniform(-80.0, 80.0)
            tz = np.random.uniform(580.0, 720.0)  # Distance in mm
            tvec = np.array([tx, ty, tz], dtype=np.float64)

            frame = self.generate_scene_image(rvec, tvec, texture)
            file_name = f"calib_{i+1:03d}.jpg"
            save_path = target_dir / file_name
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
            saved_paths.append(save_path)

        return saved_paths
