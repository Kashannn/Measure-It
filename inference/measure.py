#!/usr/bin/env python3
"""One Command: Image to Millimetres Optical Metrology Tool."""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
from inference.pipeline import MetrologyPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure-It: Optical Metrology CLI (One command, image to millimetres)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input photograph to measure.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference/diagnostics/measurement_overlay.jpg",
        help="Path to save the output annotated visual overlay image.",
    )
    parser.add_argument(
        "--no_undistort",
        action="store_true",
        help="Bypass camera undistortion (used for ablation experiments).",
    )
    parser.add_argument(
        "--camera_params",
        type=str,
        default="calibration/camera_params.json",
        help="Path to camera calibration JSON file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/checkpoints/best_model.pth",
        help="Path to trained segmentation model checkpoint.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results strictly in machine-readable JSON format.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.image)

    if not input_path.exists():
        err_payload = {
            "status": "ERROR",
            "error_message": f"Input image does not exist: {input_path}",
        }
        if args.json:
            print(json.dumps(err_payload, indent=2))
        else:
            print(f"[!] Error: {err_payload['error_message']}")
        sys.exit(1)

    # Initialize Pipeline
    pipeline = MetrologyPipeline(
        camera_params_path=args.camera_params,
        weights_path=args.weights,
    )

    # Process image
    output = pipeline.process_image(
        image_input=input_path,
        apply_undistort=not args.no_undistort,
    )

    # Save overlay image
    out_overlay_path = Path(args.output)
    out_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if output.overlay_image is not None:
        cv2.imwrite(str(out_overlay_path), output.overlay_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    result_dict = output.to_dict()
    result_dict["overlay_saved_to"] = str(out_overlay_path)
    result_dict["input_image"] = str(input_path)
    result_dict["undistortion_applied"] = not args.no_undistort

    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print("=" * 72)
        print("           Measure-It: Single-Command Metrology Result           ")
        print("=" * 72)
        print(f"  Input Image          : {input_path}")
        print(f"  Status               : {output.status}")
        print(f"  Undistortion Applied : {not args.no_undistort}")

        if output.status == "SUCCESS":
            print("-" * 72)
            print(f"  Width (mm)           : {output.width_mm:.2f} mm")
            print(f"  Height (mm)          : {output.height_mm:.2f} mm")
            print(f"  Model Confidence     : {output.confidence:.3f}")
            print(f"  Spatial Scale (PPM)  : {output.pixels_per_mm:.2f} px/mm")
            print("-" * 72)
            print(f"[+] Output overlay saved to: {out_overlay_path}")
            print("=" * 72)
            sys.exit(0)
        elif output.status == "OBJECT_NOT_FOUND":
            print("-" * 72)
            print(f"  [!] Rejection Reason : {output.error_message}")
            print(f"  Spatial Scale (PPM)  : {output.pixels_per_mm} px/mm")
            print(f"[+] Diagnostic overlay saved to: {out_overlay_path}")
            print("=" * 72)
            sys.exit(2)
        elif output.status == "REFERENCE_NOT_FOUND":
            print("-" * 72)
            print(f"  [!] Rejection Reason : {output.error_message}")
            print(f"[+] Diagnostic overlay saved to: {out_overlay_path}")
            print("=" * 72)
            sys.exit(3)
        else:
            print("-" * 72)
            print(f"  [!] Failure Error    : {output.error_message}")
            print("=" * 72)
            sys.exit(1)


if __name__ == "__main__":
    main()
