"""End-to-end metrology pipeline connecting undistortion, segmentation, and scaling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import yaml

from calibration.calibrator import CameraParameters
from models.model import build_model
from .detector import ReferenceDetector, ReferenceDetectionResult
from .geometry import MetricContourExtractor, MetricMeasurementResult
from .guardrails import InputGuardrails
from .visualizer import MetrologyVisualizer


@dataclass
class MeasurementOutput:
    """Structured metrology output format."""

    status: str  # "SUCCESS", "OBJECT_NOT_FOUND", "REFERENCE_NOT_FOUND", "ERROR"
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    confidence: Optional[float] = None
    pixels_per_mm: Optional[float] = None
    error_message: Optional[str] = None
    overlay_image: Optional[np.ndarray] = None
    undistorted_image: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Serializes clean JSON payload."""
        return {
            "status": self.status,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "confidence": self.confidence,
            "pixels_per_mm": self.pixels_per_mm,
            "error_message": self.error_message,
        }


class MetrologyPipeline:
    """Production optical metrology pipeline taking one raw image to metric millimetres."""

    def __init__(
        self,
        camera_params_path: Union[str, Path] = "calibration/camera_params.json",
        weights_path: Union[str, Path] = "models/checkpoints/best_model.pth",
        config_path: Union[str, Path] = "models/config.yaml",
        device: Optional[str] = None,
    ):
        # 1. Load Calibration Parameters (K and D)
        self.params_path = Path(camera_params_path)
        if not self.params_path.exists():
            raise FileNotFoundError(f"Calibration parameters missing: {self.params_path}")
        self.camera_params = CameraParameters.load_json(self.params_path)
        self.K = self.camera_params.camera_matrix
        self.D = self.camera_params.distortion_coefficients

        # 2. Select Device & Load Neural Segmentation Model
        self.device = torch.device(
            device
            if device
            else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        )

        cfg_p = Path(config_path)
        with open(cfg_p, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.model = build_model(self.cfg).to(self.device)
        w_path = Path(weights_path)
        if not w_path.exists():
            raise FileNotFoundError(f"Model checkpoint missing: {w_path}")

        checkpoint = torch.load(w_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 3. Initialize Detectors, Visualizers, and Guardrails
        self.ref_detector = ReferenceDetector(marker_size_mm=50.0)
        self.contour_extractor = MetricContourExtractor(min_area_px=5000)
        self.visualizer = MetrologyVisualizer()
        self.guardrails = InputGuardrails()

        # Preprocessing transforms for model input
        h_in = self.cfg.get("input_height", 512)
        w_in = self.cfg.get("input_width", 512)
        self.transform = A.Compose(
            [
                A.Resize(h_in, w_in),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Removes radial and tangential lens distortion."""
        h, w = image.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 0.0, (w, h))
        return cv2.undistort(image, self.K, self.D, None, new_K)

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        apply_undistort: bool = True,
    ) -> MeasurementOutput:
        """Executes full pipeline from image to millimetres.

        ========================================================================
        STEP-BY-STEP FLOW (Question K1 & K2 in Loom Viva)
        ========================================================================
        Step 1: Lens Undistortion (OpenCV calibrateCamera intrinsics)
        Step 2: Reference Fiducial Detection (ArUco DICT_4X4_50 -> PPM scale)
        Step 3: Neural Segmentation (DeepLabV3+ with ASPP head)
        Step 4: Geometric Metrology (cv2.minAreaRect -> mm dimensions)
        Step 5: Error Case Guardrails (Object Not Found, Reference Not Found)
        """
        # Load Image
        if isinstance(image_input, (str, Path)):
            raw_img = cv2.imread(str(image_input))
            if raw_img is None:
                return MeasurementOutput(
                    status="ERROR",
                    error_message=f"FILE_NOT_FOUND: Could not load image from {image_input}",
                )
        else:
            raw_img = image_input.copy()

        orig_h, orig_w = raw_img.shape[:2]

        # STEP 1: Lens Undistortion (K2 Answer Location)
        if apply_undistort:
            processed_img = self.undistort_image(raw_img)
        else:
            processed_img = raw_img

        # STEP 2: Detect Reference Object (ArUco Fiducial)
        ref_res = self.ref_detector.detect(processed_img)

        # Handle CASE 2: Reference Not Found
        if not ref_res.found:
            overlay = self.visualizer.render_overlay(
                processed_img,
                None,
                MetricMeasurementResult(found=False),
                ref_res,
                confidence=0.0,
            )
            return MeasurementOutput(
                status="REFERENCE_NOT_FOUND",
                error_message=ref_res.error_message,
                overlay_image=overlay,
                undistorted_image=processed_img,
            )

        # STEP 2b: Defensive Rejection Pre-Check (Lighting, Blur, Severe Tilt Skew)
        pre_guard = self.guardrails.check_pre_inference(processed_img, ref_res.tilt_skew_ratio)
        if not pre_guard.passed:
            overlay = self.visualizer.render_overlay(
                processed_img,
                None,
                MetricMeasurementResult(found=False),
                ref_res,
                confidence=0.0,
            )
            return MeasurementOutput(
                status="UNTRUSTWORTHY_INPUT_REJECTED",
                error_message=pre_guard.reason,
                pixels_per_mm=round(ref_res.pixels_per_mm, 2) if ref_res.pixels_per_mm else None,
                overlay_image=overlay,
                undistorted_image=processed_img,
            )

        # STEP 3: Semantic Segmentation Inference
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        tensor_in = self.transform(image=rgb_img)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.sigmoid(logits)

        # Upsample 512x512 prediction back to original full sensor resolution
        probs_512 = probs.squeeze().cpu().numpy()
        probs_full = cv2.resize(probs_512, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask_full = (probs_full >= 0.50).astype(np.uint8) * 255

        # Suppress ArUco marker footprint from object mask to avoid cross-contamination
        if ref_res.found and ref_res.corners is not None:
            marker_hull = cv2.convexHull(ref_res.corners.reshape((-1, 1, 2)).astype(np.int32))
            cv2.fillPoly(binary_mask_full, [marker_hull], 0)

        # Calculate foreground confidence
        fg = binary_mask_full > 0
        confidence = float(np.mean(probs_full[fg])) if np.any(fg) else 0.0

        # STEP 4: Geometric Dimension Extraction (K1 Answer Location)
        metric_res = self.contour_extractor.extract_dimensions(
            binary_mask_full, pixels_per_mm=ref_res.pixels_per_mm
        )

        # Handle CASE 1: Object Not Found
        if not metric_res.found:
            overlay = self.visualizer.render_overlay(
                processed_img,
                binary_mask_full,
                metric_res,
                ref_res,
                confidence=confidence,
            )
            return MeasurementOutput(
                status="OBJECT_NOT_FOUND",
                error_message=metric_res.error_message,
                confidence=round(confidence, 3),
                pixels_per_mm=round(ref_res.pixels_per_mm, 2),
                overlay_image=overlay,
                undistorted_image=processed_img,
                binary_mask=binary_mask_full,
            )

        # STEP 4b: Defensive Rejection Post-Check (Low Model Confidence, Anomalous Geometry)
        post_guard = self.guardrails.check_post_inference(
            confidence, metric_res.width_mm, metric_res.height_mm
        )
        if not post_guard.passed:
            overlay = self.visualizer.render_overlay(
                processed_img,
                binary_mask_full,
                metric_res,
                ref_res,
                confidence=confidence,
            )
            return MeasurementOutput(
                status="UNTRUSTWORTHY_INPUT_REJECTED",
                error_message=post_guard.reason,
                width_mm=round(metric_res.width_mm, 2),
                height_mm=round(metric_res.height_mm, 2),
                confidence=round(confidence, 3),
                pixels_per_mm=round(ref_res.pixels_per_mm, 2),
                overlay_image=overlay,
                undistorted_image=processed_img,
                binary_mask=binary_mask_full,
            )

        # STEP 5: Render Successful Metrology Overlay
        overlay = self.visualizer.render_overlay(
            processed_img,
            binary_mask_full,
            metric_res,
            ref_res,
            confidence=confidence,
        )

        return MeasurementOutput(
            status="SUCCESS",
            width_mm=metric_res.width_mm,
            height_mm=metric_res.height_mm,
            confidence=round(confidence, 3),
            pixels_per_mm=round(ref_res.pixels_per_mm, 2),
            overlay_image=overlay,
            undistorted_image=processed_img,
            binary_mask=binary_mask_full,
        )
