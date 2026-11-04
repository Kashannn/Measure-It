"""Visual overlay generator for metrology annotations and dimension callouts."""

from typing import Optional
import cv2
import numpy as np

from .detector import ReferenceDetectionResult
from .geometry import MetricMeasurementResult


class MetrologyVisualizer:
    """Renders industrial diagnostic overlays with dimension arrows, bounding boxes, and HUD."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render_overlay(
        self,
        image: np.ndarray,
        binary_mask: Optional[np.ndarray],
        metric_result: MetricMeasurementResult,
        ref_result: ReferenceDetectionResult,
        confidence: float,
    ) -> np.ndarray:
        """Renders comprehensive metrology overlay image."""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        # 1. Overlay semi-transparent mask if object found
        if binary_mask is not None and metric_result.found:
            colored_mask = np.zeros_like(overlay)
            colored_mask[binary_mask > 0] = [0, 220, 100]  # Emerald green
            cv2.addWeighted(colored_mask, 0.35, overlay, 0.65, 0, overlay)

            # Draw outer contour boundary
            if metric_result.contour is not None:
                cv2.drawContours(overlay, [metric_result.contour], -1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw oriented bounding box
            if metric_result.box_corners is not None:
                cv2.polylines(overlay, [metric_result.box_corners], True, (0, 255, 255), 2, cv2.LINE_AA)

                # Annotate dimension callout near object center
                cx, cy = int(metric_result.center_px[0]), int(metric_result.center_px[1])
                dim_text = f"{metric_result.width_mm:.1f} x {metric_result.height_mm:.1f} mm"
                cv2.putText(overlay, dim_text, (cx - 70, cy - 10), self.font, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, dim_text, (cx - 70, cy - 10), self.font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Draw ArUco reference marker if detected
        if ref_result.found and ref_result.corners is not None:
            pts = ref_result.corners.astype(np.int32)
            cv2.polylines(overlay, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            p0 = pts[0]
            ref_label = f"Ref ArUco #{ref_result.marker_id} ({ref_result.pixels_per_mm:.2f} px/mm)"
            cv2.putText(overlay, ref_label, (p0[0], p0[1] - 8), self.font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # 3. Render Top HUD Status Banner
        is_success = metric_result.found and ref_result.found
        hud_h = 50
        hud_bg = (25, 120, 25) if is_success else (25, 25, 160)
        cv2.rectangle(overlay, (0, 0), (w, hud_h), hud_bg, -1)

        if is_success:
            hud_text = (
                f"MEASURE-IT: OK | Width: {metric_result.width_mm:.2f} mm | "
                f"Height: {metric_result.height_mm:.2f} mm | "
                f"Confidence: {confidence:.3f} | Scale: {ref_result.pixels_per_mm:.2f} px/mm"
            )
        else:
            reason = metric_result.error_message or ref_result.error_message or "Unknown failure"
            hud_text = f"MEASURE-IT: REFUSED / ERROR | {reason}"

        cv2.putText(overlay, hud_text, (20, 32), self.font, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        return overlay
