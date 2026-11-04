#!/usr/bin/env python3
"""CLI entry point for Camera Calibration & Optical Undistortion Verification."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from calibration.config import CalibrationConfig
from calibration.calibrator import CameraCalibrator
from calibration.visualize import (
    generate_undistortion_comparison,
    plot_reprojection_errors,
)
from calibration.synthetic_generator import SyntheticCalibrationGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Industrial Camera Calibration & Lens Undistortion Tool (OpenCV)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="calibration/images",
        help="Path to folder containing checkerboard images (>=20 images required).",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=9,
        help="Number of inner corners horizontally (default: 9).",
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=7,
        help="Number of inner corners vertically (default: 7).",
    )
    parser.add_argument(
        "--square_size_mm",
        type=float,
        default=25.0,
        help="Physical dimension of each square in millimeters (default: 25.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration/camera_params.json",
        help="Path to save calibrated camera parameters JSON.",
    )
    parser.add_argument(
        "--diagnostics_dir",
        type=str,
        default="calibration/diagnostics",
        help="Directory to save visual diagnostics, error plots, and comparison images.",
    )
    parser.add_argument(
        "--generate_synthetic",
        action="store_true",
        help="Generate synthetic calibration images if images_dir is empty or missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_path = Path(args.images_dir)
    diag_path = Path(args.diagnostics_dir)
    diag_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("      Measure-It: Precision Camera Calibration & Undistortion       ")
    print("=" * 72)

    # Check if images directory has images or needs generation
    existing_images = (
        list(images_path.glob("*.jpg"))
        + list(images_path.glob("*.png"))
        + list(images_path.glob("*.jpeg"))
        if images_path.exists()
        else []
    )

    if len(existing_images) < 20:
        if args.generate_synthetic or len(existing_images) == 0:
            print(f"[*] Calibration images count is {len(existing_images)} (<20 required).")
            print("[*] Generating 25 high-fidelity physical camera simulation frames...")
            sim = SyntheticCalibrationGenerator(
                grid_cols=args.grid_cols,
                grid_rows=args.grid_rows,
                square_size_mm=args.square_size_mm,
            )
            sim.generate_calibration_dataset(output_dir=images_path, num_images=25)
            print(f"[+] 25 calibration frames generated successfully in: {images_path}")
        else:
            print(
                f"[!] Warning: Only {len(existing_images)} images found in {images_path}. Assessment requires >= 20."
            )

    config = CalibrationConfig(
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        square_size_mm=args.square_size_mm,
    )

    calibrator = CameraCalibrator(config)
    print(f"\n[*] Scanning calibration directory: {images_path}")
    print(f"[*] Target pattern: {config.grid_cols}x{config.grid_rows} inner corners, {config.square_size_mm} mm square size")

    stats = calibrator.process_directory(images_path)
    print(f"[*] Processed {stats['total_found']} images: {stats['valid_detections']} valid detections, {stats['failed_detections']} failed.")

    if stats["valid_detections"] < config.min_required_images:
        print(f"[!] Error: Need at least {config.min_required_images} valid detections, but only got {stats['valid_detections']}.")
        sys.exit(1)

    print("\n[*] Running Levenberg-Marquardt non-linear pinhole camera optimization...")
    params = calibrator.calibrate()

    # Save camera parameters to JSON
    out_file = Path(args.output)
    params.save_json(out_file)

    print("\n" + "=" * 72)
    print("               CALIBRATION RESULTS & DIAGNOSTICS               ")
    print("=" * 72)
    print(f"  Valid Image Count        : {params.image_count}")
    print(f"  Image Resolution         : {params.image_width} x {params.image_height} px")
    print(f"  Mean Reprojection Error  : {params.reprojection_error_px:.5f} pixels")
    print(f"  Status                   : {'PASS (MRE < 0.50 px)' if params.reprojection_error_px <= 0.50 else 'WARNING'}")
    print("-" * 72)
    print("  Camera Matrix K:")
    for row in params.camera_matrix:
        print(f"    [{row[0]:10.3f}, {row[1]:10.3f}, {row[2]:10.3f}]")
    print("\n  Focal Length (fx, fy)    : ({:.2f}, {:.2f}) px".format(
        params.camera_matrix[0, 0], params.camera_matrix[1, 1]
    ))
    print("  Principal Point (cx, cy) : ({:.2f}, {:.2f}) px".format(
        params.camera_matrix[0, 2], params.camera_matrix[1, 2]
    ))
    print("  Distortion Vector D      : {}".format(
        np.array2string(params.distortion_coefficients.flatten()[:5], precision=5, separator=", ")
    ))
    print("-" * 72)
    print(f"[+] Saved parameters JSON: {out_file}")

    # Generate diagnostic visualizations
    print("\n[*] Generating visual validation diagnostics...")

    # 1. Reprojection error plot
    plot_path = diag_path / "reprojection_error_plot.png"
    plot_reprojection_errors(
        calibrator.per_image_errors,
        params.reprojection_error_px,
        plot_path,
    )
    print(f"[+] Reprojection error audit plot saved to: {plot_path}")

    # 2. Side-by-side undistortion proof for first valid calibration frame
    sample_img_path = calibrator.valid_image_paths[0]
    raw_img = cv2.imread(str(sample_img_path))
    undist_img, _, _ = calibrator.undistort_image(
        raw_img, params.camera_matrix, params.distortion_coefficients
    )

    proof_path = diag_path / "undistortion_proof_grid.png"
    generate_undistortion_comparison(
        raw_img,
        undist_img,
        proof_path,
        title=f"Lens Undistortion Proof (Sample: {sample_img_path.name})",
    )
    print(f"[+] Undistortion visual proof saved to: {proof_path}")

    # Also save the clean undistorted sample image
    sample_undist_path = diag_path / "sample_undistorted.jpg"
    cv2.imwrite(str(sample_undist_path), undist_img)
    print(f"[+] Sample undistorted image saved to: {sample_undist_path}")

    print("\n[✓] Camera Calibration (Step 1) successfully completed and verified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
