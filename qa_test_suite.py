#!/usr/bin/env python3
"""Measure-It: Comprehensive Quality Assurance (QA) & Smoke Test Suite.

Executes all pipeline components, checks exit codes, and validates all metrology
operations end-to-end to guarantee rock-solid stability before candidate submission.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Runs a command and returns (exit_code, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    print("=" * 80)
    print("           Measure-It: Comprehensive Quality Assurance (QA) Suite           ")
    print("=" * 80)

    qa_results: List[Dict] = []

    # -------------------------------------------------------------------------
    # TEST 1: Camera Calibration
    # -------------------------------------------------------------------------
    print("\n[+] [QA-01] Testing Camera Calibration Pipeline...")
    ret, out, err = run_cmd([
        sys.executable,
        "calibration/calibrate.py",
        "--images_dir",
        "calibration/images",
        "--output",
        "calibration/camera_params.json",
    ])
    status = "PASS" if (ret == 0 and Path("calibration/camera_params.json").exists()) else "FAIL"
    qa_results.append({"test": "QA-01: Camera Calibration", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 2: Dataset Integrity Audit
    # -------------------------------------------------------------------------
    print("\n[+] [QA-02] Testing Dataset Split & Mask Integrity Audit...")
    ret, out, err = run_cmd([sys.executable, "dataset/verify_dataset.py"])
    status = "PASS" if ret == 0 else "FAIL"
    qa_results.append({"test": "QA-02: Dataset Integrity Audit", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 3: Isolated Test Split Model Evaluation
    # -------------------------------------------------------------------------
    print("\n[+] [QA-03] Testing Test Split Segmentation Model Evaluation...")
    ret, out, err = run_cmd([
        sys.executable,
        "models/evaluate.py",
        "--weights",
        "models/checkpoints/best_model.pth",
        "--split",
        "dataset/splits/test.txt",
    ])
    status = "PASS" if ret == 0 else "FAIL"
    qa_results.append({"test": "QA-03: Model Evaluation (test split)", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 4: Single-Command CLI (Normal Case -> Exit Code 0)
    # -------------------------------------------------------------------------
    print("\n[+] [QA-04] Testing Single-Command Root CLI (measure.py)...")
    ret, out, err = run_cmd([
        sys.executable,
        "measure.py",
        "--image",
        "dataset/images/img_0073.jpg",
        "--output",
        "inference/diagnostics/qa_overlay.jpg",
    ])
    overlay_exists = Path("inference/diagnostics/qa_overlay.jpg").exists()
    status = "PASS" if (ret == 0 and overlay_exists) else "FAIL"
    qa_results.append({"test": "QA-04: Single-Command CLI (Normal)", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} -> Status: {status} (Overlay generated: {overlay_exists})")

    # -------------------------------------------------------------------------
    # TEST 5: Single-Command CLI JSON Output Mode
    # -------------------------------------------------------------------------
    print("\n[+] [QA-05] Testing JSON Serialization Mode (--json)...")
    ret, out, err = run_cmd([
        sys.executable,
        "measure.py",
        "--image",
        "dataset/images/img_0073.jpg",
        "--json",
    ])
    json_valid = False
    try:
        # Extract json part from stdout
        lines = out.strip().split("\n")
        json_str = "\n".join([line for line in lines if not line.startswith("/") and not line.startswith("UserWarning")])
        data = json.loads(json_str)
        if data.get("status") == "SUCCESS" and "width_mm" in data:
            json_valid = True
    except Exception:
        json_valid = False
    status = "PASS" if (ret == 0 and json_valid) else "FAIL"
    qa_results.append({"test": "QA-05: JSON Output Format", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} -> Status: {status} (Valid JSON output: {json_valid})")

    # -------------------------------------------------------------------------
    # TEST 6: Edge Case 1 — Missing File (Expected Exit Code 1)
    # -------------------------------------------------------------------------
    print("\n[+] [QA-06] Testing Edge Case 1: Missing Image File...")
    ret, out, err = run_cmd([
        sys.executable,
        "measure.py",
        "--image",
        "dataset/images/non_existent_file.jpg",
    ])
    status = "PASS" if ret == 1 else "FAIL"
    qa_results.append({"test": "QA-06: Missing File Handling (Code 1)", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} (Expected 1) -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 7: Edge Case 2 — Reference Not Found (Expected Exit Code 3)
    # -------------------------------------------------------------------------
    print("\n[+] [QA-07] Testing Edge Case 2: Reference Marker Missing...")
    # Crop out ArUco marker (keep only left half)
    crop_script = "import cv2; img = cv2.imread('dataset/images/img_0073.jpg'); cv2.imwrite('inference/diagnostics/qa_no_aruco.jpg', img[:, :500])"
    subprocess.run([sys.executable, "-c", crop_script], check=True)
    ret, out, err = run_cmd([
        sys.executable,
        "measure.py",
        "--image",
        "inference/diagnostics/qa_no_aruco.jpg",
    ])
    status = "PASS" if ret == 3 else "FAIL"
    qa_results.append({"test": "QA-07: Reference Not Found Handling (Code 3)", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} (Expected 3) -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 8: Edge Case 3 — Object Not Found (Expected Exit Code 2)
    # -------------------------------------------------------------------------
    print("\n[+] [QA-08] Testing Edge Case 3: Target Specimen Missing...")
    # Generate marker only canvas
    marker_only_script = (
        "import cv2, numpy as np; "
        "d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); "
        "raw = cv2.aruco.generateImageMarker(d, 0, 200); "
        "c = np.ones((720, 1280, 3), dtype=np.uint8)*230; "
        "c[100:300, 100:300] = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR); "
        "cv2.imwrite('inference/diagnostics/qa_marker_only.jpg', c)"
    )
    subprocess.run([sys.executable, "-c", marker_only_script], check=True)
    ret, out, err = run_cmd([
        sys.executable,
        "measure.py",
        "--image",
        "inference/diagnostics/qa_marker_only.jpg",
    ])
    status = "PASS" if ret == 2 else "FAIL"
    qa_results.append({"test": "QA-08: Object Not Found Handling (Code 2)", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} (Expected 2) -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 9: Defensive Rejection — Excessive Blur (Expected Exit Code 4)
    # -------------------------------------------------------------------------
    print("\n[+] [QA-09] Testing Defensive Rejection: Blurry Image...")
    blur_script = (
        "import cv2; "
        "img = cv2.imread('dataset/images/img_0073.jpg'); "
        "blurry = cv2.GaussianBlur(img, (51, 51), 0); "
        "cv2.imwrite('inference/diagnostics/qa_blur.jpg', blurry)"
    )
    subprocess.run([sys.executable, "-c", blur_script], check=True)
    ret, out, err = run_cmd([
        sys.executable,
        "measure.py",
        "--image",
        "inference/diagnostics/qa_blur.jpg",
    ])
    status = "PASS" if ret == 4 else "FAIL"
    qa_results.append({"test": "QA-09: Defensive Refusal - Blur (Code 4)", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} (Expected 4) -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 10: Digital Caliper Accuracy Benchmark (MO-1 Target)
    # -------------------------------------------------------------------------
    print("\n[+] [QA-10] Testing Caliper Accuracy Benchmarking (MO-1)...")
    ret, out, err = run_cmd([sys.executable, "measurement/benchmark_accuracy.py"])
    status = "PASS" if ret == 0 else "FAIL"
    qa_results.append({"test": "QA-10: Caliper Benchmark (MO-1 Target)", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} -> Status: {status}")

    # -------------------------------------------------------------------------
    # TEST 11: Undistortion Ablation Impact Test
    # -------------------------------------------------------------------------
    print("\n[+] [QA-11] Testing Lens Undistortion Ablation Comparative Test...")
    ret, out, err = run_cmd([sys.executable, "measurement/ablation_undistort.py"])
    status = "PASS" if ret == 0 else "FAIL"
    qa_results.append({"test": "QA-11: Undistortion Ablation", "status": status, "exit_code": ret})
    print(f"    Exit Code: {ret} -> Status: {status}")

    # -------------------------------------------------------------------------
    # Cleanup temporary QA artifacts
    # -------------------------------------------------------------------------
    for temp_p in [
        "inference/diagnostics/qa_overlay.jpg",
        "inference/diagnostics/qa_no_aruco.jpg",
        "inference/diagnostics/qa_marker_only.jpg",
        "inference/diagnostics/qa_blur.jpg",
    ]:
        p = Path(temp_p)
        if p.exists():
            p.unlink()

    # -------------------------------------------------------------------------
    # QA SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                           QA TEST SUITE SUMMARY                           ")
    print("=" * 80)
    print(f"{'Test Case':<52} | {'Exit Code':<10} | {'Status'}")
    print("-" * 80)
    all_passed = True
    for item in qa_results:
        print(f"{item['test']:<52} | {item['exit_code']:<10} | {item['status']}")
        if item["status"] != "PASS":
            all_passed = False
    print("=" * 80)

    if all_passed:
        print(" [✓] ALL 11 QA & SMOKE TESTS PASSED PERFECTLY! SYSTEM PRODUCTION READY.")
        print("=" * 80)
        sys.exit(0)
    else:
        print(" [!] ONE OR MORE TESTS FAILED.")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
