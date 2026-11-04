"""High-fidelity dataset generator for industrial metrology specimen inspection."""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Tuple, Union
import cv2
import numpy as np

from dataset.config import DatasetConfig


class MetrologyDatasetGenerator:
    """Renders realistic physical inspection images with coplanar ArUco marker and ground-truth masks."""

    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        self.w, self.h = self.config.capture_resolution

        # ArUco marker generator
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def generate_aruco_texture(self, size_px: int = 240) -> np.ndarray:
        """Draws the 50x50mm ArUco marker ID 0 with a white border."""
        # Generate raw 4x4 marker
        marker_raw = cv2.aruco.generateImageMarker(self.aruco_dict, 0, size_px)

        # Add physical white quiet zone / paper margin (35% of size)
        margin = int(size_px * 0.35)
        full_size = size_px + 2 * margin
        marker = np.ones((full_size, full_size, 3), dtype=np.uint8) * 250
        marker[margin : margin + size_px, margin : margin + size_px] = cv2.cvtColor(
            marker_raw, cv2.COLOR_GRAY2BGR
        )
        return marker

    def generate_object_texture(self, w_px: int, h_px: int) -> Tuple[np.ndarray, np.ndarray]:
        """Creates the physical target specimen: metallic / anodized industrial block with rounded corners."""
        texture = np.ones((h_px, w_px, 3), dtype=np.uint8) * 45  # Matte charcoal / anodized metal
        # Subtle texture / brushed gradient
        for y in range(h_px):
            val = int(45 + 15 * np.sin(y / 15.0))
            texture[y, :, :] = np.clip(val + np.random.normal(0, 2, (w_px, 3)), 0, 255).astype(np.uint8)

        # Chamfered/rounded corners
        radius = 8
        mask = np.zeros((h_px, w_px), dtype=np.uint8)
        cv2.rectangle(mask, (radius, 0), (w_px - radius, h_px), 255, -1)
        cv2.rectangle(mask, (0, radius), (w_px, h_px - radius), 255, -1)
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (w_px - radius, radius), radius, 255, -1)
        cv2.circle(mask, (radius, h_px - radius), radius, 255, -1)
        cv2.circle(mask, (w_px - radius, h_px - radius), radius, 255, -1)

        # Border bevel / highlight line
        cv2.rectangle(texture, (2, 2), (w_px - 3, h_px - 3), (85, 85, 90), 1)

        # Technical label print on object
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(texture, "XIS-METRIC", (w_px // 2 - 40, h_px // 2 - 8), font, 0.38, (210, 210, 215), 1, cv2.LINE_AA)
        cv2.putText(texture, "54x85.5 mm", (w_px // 2 - 38, h_px // 2 + 12), font, 0.32, (180, 180, 185), 1, cv2.LINE_AA)

        return texture, mask

    def create_surface_background(self, theme_idx: int) -> np.ndarray:
        """Generates realistic workbench surfaces."""
        bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        if theme_idx % 4 == 0:
            # Light maple/wood desk surface
            bg[:] = (205, 220, 235)  # Light wood warm tone (BGR)
            noise = np.random.normal(0, 5, bg.shape).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif theme_idx % 4 == 1:
            # Industrial light gray ESD mat
            bg[:] = (195, 195, 200)
            noise = np.random.normal(0, 6, bg.shape).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif theme_idx % 4 == 2:
            # Dark graphite workbench
            bg[:] = (75, 75, 80)
            noise = np.random.normal(0, 4, bg.shape).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            # Clean off-white optical measurement surface
            bg[:] = (235, 238, 240)
            noise = np.random.normal(0, 3, bg.shape).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add subtle lighting gradient (vignette / ambient falloff)
        y, x = np.ogrid[: self.h, : self.w]
        center_x, center_y = self.w / 2.0, self.h / 2.0
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        vignette = 1.0 - 0.15 * (dist / np.max(dist))
        bg = (bg * vignette[:, :, None]).astype(np.uint8)
        return bg

    def paste_rotated_patch(
        self,
        canvas: np.ndarray,
        canvas_mask: np.ndarray,
        patch: np.ndarray,
        patch_mask: np.ndarray,
        center: Tuple[int, int],
        angle_deg: float,
        is_target_object: bool = False,
    ) -> None:
        """Pastes a rotated patch onto the canvas and updates the binary mask."""
        ph, pw = patch.shape[:2]
        cx, cy = center

        # Rotation matrix around patch center
        M = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle_deg, 1.0)
        # Compute new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((ph * sin) + (pw * cos))
        new_h = int((ph * cos) + (pw * sin))
        M[0, 2] += (new_w / 2.0) - pw / 2.0
        M[1, 2] += (new_h / 2.0) - ph / 2.0

        border_color = (0, 0, 0) if is_target_object else (250, 250, 250)
        rotated_patch = cv2.warpAffine(patch, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=border_color)
        rotated_mask = cv2.warpAffine(patch_mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)

        # Place onto canvas
        x0 = int(cx - new_w // 2)
        y0 = int(cy - new_h // 2)

        # Clip bounds
        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)

        dst_x1 = min(self.w, x0 + new_w)
        dst_y1 = min(self.h, y0 + new_h)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return

        patch_crop = rotated_patch[src_y0:src_y1, src_x0:src_x1]
        mask_crop = rotated_mask[src_y0:src_y1, src_x0:src_x1]

        # Soft contact shadow
        shadow_mask = cv2.GaussianBlur(mask_crop, (21, 21), 0)
        canvas_slice = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(
            canvas_slice.astype(np.float32) * (1.0 - 0.25 * (shadow_mask[:, :, None] / 255.0)), 0, 255
        ).astype(np.uint8)

        # Alpha composite patch
        m_3c = cv2.merge([mask_crop, mask_crop, mask_crop])
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.where(m_3c > 128, patch_crop, canvas[dst_y0:dst_y1, dst_x0:dst_x1])

        if is_target_object:
            canvas_mask[dst_y0:dst_y1, dst_x0:dst_x1] = np.where(
                mask_crop > 128, 255, canvas_mask[dst_y0:dst_y1, dst_x0:dst_x1]
            )

    def generate_single_sample(self, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generates one high-resolution scene with target object, ArUco reference, and ground truth."""
        rng = np.random.RandomState(seed)

        # Pixel scale: ArUco 50mm is roughly 220-280 pixels at typical working distance
        # Let ppm (pixels per mm) be ~ 4.8 px/mm
        ppm = rng.uniform(4.2, 5.4)

        aruco_px = int(self.config.aruco_size_mm * ppm)
        obj_w_px = int(self.config.object_nominal_width_mm * ppm)
        obj_h_px = int(self.config.object_nominal_height_mm * ppm)

        aruco_texture = self.generate_aruco_texture(size_px=aruco_px)
        aruco_mask = np.ones(aruco_texture.shape[:2], dtype=np.uint8) * 255

        obj_texture, obj_mask = self.generate_object_texture(obj_w_px, obj_h_px)

        canvas = self.create_surface_background(seed)
        canvas_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # Designated inspection zone for target object (left / central zone)
        obj_cx = rng.randint(450, 950)
        obj_cy = rng.randint(350, 750)
        obj_rot = rng.uniform(0.0, 360.0)

        # Designated metric fiducial zone (coplanar right side of plate)
        aruco_cx = rng.randint(1300, 1650)
        aruco_cy = rng.randint(350, 750)
        aruco_rot = rng.uniform(-10.0, 10.0)

        # Paste ArUco marker (is_target_object=False)
        self.paste_rotated_patch(
            canvas, canvas_mask, aruco_texture, aruco_mask, (aruco_cx, aruco_cy), aruco_rot, is_target_object=False
        )

        # Paste target object (is_target_object=True -> updates canvas_mask)
        self.paste_rotated_patch(
            canvas, canvas_mask, obj_texture, obj_mask, (obj_cx, obj_cy), obj_rot, is_target_object=True
        )

        meta = {
            "seed": seed,
            "ppm": round(ppm, 4),
            "object_nominal_mm": [self.config.object_nominal_width_mm, self.config.object_nominal_height_mm],
            "object_center_px": [obj_cx, obj_cy],
            "object_rotation_deg": round(obj_rot, 2),
            "aruco_center_px": [aruco_cx, aruco_cy],
            "aruco_rotation_deg": round(aruco_rot, 2),
        }

        return canvas, canvas_mask, meta

    def build_dataset(self, num_samples: int = 80) -> None:
        """Generates all 80 dataset images and masks."""
        self.config.images_dir.mkdir(parents=True, exist_ok=True)
        self.config.labels_dir.mkdir(parents=True, exist_ok=True)

        annotations = {}

        print(f"[*] Generating {num_samples} physical metrology inspection images and masks...")
        for i in range(1, num_samples + 1):
            file_base = f"img_{i:04d}"
            img_path = self.config.images_dir / f"{file_base}.jpg"
            mask_path = self.config.labels_dir / f"{file_base}.png"

            image, mask, meta = self.generate_single_sample(seed=1000 + i)

            # Save JPEG for photo, lossless PNG for binary mask
            cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 96])
            cv2.imwrite(str(mask_path), mask)

            annotations[f"{file_base}.jpg"] = meta

            if i % 20 == 0 or i == num_samples:
                print(f"  [+] Generated {i}/{num_samples} samples ({img_path.name})")

        # Save annotations JSON
        anno_file = self.config.dataset_root / "annotations.json"
        with open(anno_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

        print(f"[✓] Saved dataset annotations to: {anno_file}")


if __name__ == "__main__":
    generator = MetrologyDatasetGenerator()
    generator.build_dataset(num_samples=80)
