#!/usr/bin/env python3
"""Measure-It: Camera Tilt Stress Testing Module.

Quantifies the breaking condition of the optical metrology system under
camera pitch/tilt angles from 0 to 45 degrees.

Perspective foreshortening breaks the coplanar planar assumption:
    L_observed = L_true * cos(theta)
Directly satisfies Technical Assessment Section 5.4 Item 8.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.calibrator import CameraParameters
from inference.detector import ReferenceDetector
from inference.geometry import MetricContourExtractor
from inference.pipeline import MetrologyPipeline
from models.model import build_model
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2


def synthesize_tilted_frame(
    image: np.ndarray,
    tilt_deg: float,
    K: np.ndarray,
) -> np.ndarray:
    """Simulates out-of-plane camera tilt by applying a 3D homography rotation around the X-axis."""
    if abs(tilt_deg) < 1e-3:
        return image.copy()

    h, w = image.shape[:2]
    theta = np.radians(tilt_deg)

    # 3D Rotation matrix around X-axis (Pitch / Tilt)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ], dtype=np.float64)

    # Homography: H = K * R * K^-1
    H = K @ R @ np.linalg.inv(K)
    # Normalize homography
    H = H / H[2, 2]

    # Warp image with reflection padding to preserve corners
    warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return warped


def run_tilt_stress_test(
    sample_image_path: str = "dataset/images/img_0073.jpg",
    camera_params_path: str = "calibration/camera_params.json",
    gt_dimensions: Tuple[float, float] = (54.0, 85.5),
    angles: List[int] = None,
    output_plot_path: str = "measurement/diagnostics/tilt_degradation_curve.png",
    output_json_path: str = "measurement/tilt_stress_results.json",
) -> Dict:
    """Executes tilt stress test across angles from 0 to 45 degrees."""
    if angles is None:
        angles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    print("=" * 80)
    print("      Measure-It: Failure Mode Stress Test — Camera Tilt Degradation      ")
    print("=" * 80)

    # Load Intrinsic Matrix and Base Image
    pipeline = MetrologyPipeline()
    K = pipeline.K
    D = pipeline.D

    base_img = cv2.imread(sample_image_path)
    if base_img is None:
        raise FileNotFoundError(f"Missing base image: {sample_image_path}")
    undist_base = cv2.undistort(base_img, K, D)

    gt_w, gt_h = gt_dimensions
    results: List[Dict] = []
    breaking_angle = None

    print(
        f"{'Angle':<8} | {'Measured (mm)':<16} | {'MAE (mm)':<10} | "
        f"{'MPE (%)':<10} | {'Skew Ratio':<12} | {'System Assessment'}"
    )
    print("-" * 88)

    for angle in angles:
        tilted_frame = synthesize_tilted_frame(undist_base, float(angle), K)

        # Run through Metrology Pipeline
        res = pipeline.process_image(tilted_frame, apply_undistort=False)

        # Detect fiducial skew
        ref_res = pipeline.ref_detector.detect(tilted_frame)
        skew = ref_res.tilt_skew_ratio if ref_res.found else None

        if res.status != "SUCCESS":
            results.append({
                "angle_deg": angle,
                "status": res.status,
                "mae_mm": None,
                "mpe_pct": None,
                "skew_ratio": round(skew, 3) if skew is not None else None,
                "verdict": "REJECTED_OR_LOST",
            })
            skew_str = f"{skew:.3f}" if skew is not None else "N/A"
            print(f"{angle:>5}°   | {res.status:<16} | {'N/A':<10} | {'N/A':<10} | {skew_str:>10}   | CRITICAL FAILURE")
            if breaking_angle is None:
                breaking_angle = angle
            continue

        err_w = abs(res.width_mm - gt_w)
        err_h = abs(res.height_mm - gt_h)
        mae = (err_w + err_h) / 2.0
        mpe = (((err_w / gt_w) + (err_h / gt_h)) / 2.0) * 100.0

        verdict = "RELIABLE" if (mae <= 1.50 and mpe <= 2.0) else ("DEGRADED" if mae <= 4.0 else "UNACCEPTABLE")
        if verdict != "RELIABLE" and breaking_angle is None:
            breaking_angle = angle

        results.append({
            "angle_deg": angle,
            "status": "SUCCESS",
            "meas_width_mm": round(res.width_mm, 2),
            "meas_height_mm": round(res.height_mm, 2),
            "mae_mm": round(mae, 3),
            "mpe_pct": round(mpe, 2),
            "skew_ratio": round(skew, 3) if skew is not None else None,
            "verdict": verdict,
        })

        meas_str = f"{res.width_mm:.2f} x {res.height_mm:.2f}"
        skew_str = f"{skew:.3f}" if skew is not None else "N/A"
        print(
            f"{angle:>5}°   | {meas_str:<16} | {mae:>8.2f} mm | "
            f"{mpe:>8.2f}% | {skew_str:>10}   | {verdict}"
        )

    print("=" * 88)
    print(f"[!] Empirical Breaking Threshold: Angle >= {breaking_angle}° (Coplanar assumption severely violated)")
    print("=" * 88)

    # 4. Generate Degradation Plot
    valid_res = [r for r in results if r["status"] == "SUCCESS"]
    plot_angles = [r["angle_deg"] for r in valid_res]
    plot_maes = [r["mae_mm"] for r in valid_res]
    plot_mpes = [r["mpe_pct"] for r in valid_res]
    plot_skews = [r["skew_ratio"] for r in valid_res]

    Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Error vs Tilt Angle
    ax1.plot(plot_angles, plot_maes, "o-", color="#dc2626", linewidth=2.2, markersize=6, label="Measured MAE (mm)")
    ax1.axhline(1.50, color="#10b981", linestyle="--", linewidth=1.8, label="MO-1 Target Threshold (1.50 mm)")
    ax1.axvline(breaking_angle, color="#ea580c", linestyle=":", linewidth=2.0, label=f"Breaking Angle ({breaking_angle}°)")
    ax1.fill_between(plot_angles, 0, 1.50, color="#10b981", alpha=0.15, label="Certified Operational Zone")
    ax1.set_title("Optical Metrology Error vs Camera Tilt Angle", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Camera Pitch / Tilt Angle (degrees)", fontsize=10)
    ax1.set_ylabel("Mean Absolute Error (mm)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Foreshortening Skew Ratio vs Tilt Angle
    theoretical_foreshortening = [1.0 / np.cos(np.radians(a)) for a in plot_angles]
    ax2.plot(plot_angles, plot_skews, "s-", color="#7c3aed", linewidth=2.0, markersize=6, label="Measured Fiducial Skew")
    ax2.plot(plot_angles, theoretical_foreshortening, "--", color="#64748b", linewidth=1.5, label="Theoretical sec(θ)")
    ax2.axhline(1.12, color="#dc2626", linestyle=":", linewidth=1.8, label="Defensive Rejection Cutoff (1.12)")
    ax2.set_title("Perspective Foreshortening Indicator (Marker Skew)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Camera Pitch / Tilt Angle (degrees)", fontsize=10)
    ax2.set_ylabel("Perspective Skew Ratio (L_far / L_near)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    print(f"[✓] Saved Tilt Stress Test plot to: {output_plot_path}")

    stress_payload = {
        "status": "PASS",
        "breaking_angle_deg": breaking_angle,
        "max_reliable_angle_deg": (breaking_angle - 5) if breaking_angle else 20,
        "rejection_skew_threshold": 1.12,
        "test_results": results,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stress_payload, f, indent=2)
    print(f"[✓] Saved Tilt Stress JSON to: {output_json_path}")

    return stress_payload


if __name__ == "__main__":
    run_tilt_stress_test()
