#!/usr/bin/env python3
"""Measure-It: Caliper Accuracy Benchmarking Module.

Evaluates system measurements against physical digital caliper ground truth
across 10+ held-out instances to verify Measurable Objective 1 (MO-1):
    - Target MAE <= 1.50 mm
    - Target MPE <= 2.0%
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.pipeline import MetrologyPipeline


def run_caliper_benchmark(
    annotations_path: str = "dataset/annotations.json",
    test_split_path: str = "dataset/splits/test.txt",
    val_split_path: str = "dataset/splits/val.txt",
    output_plot_path: str = "measurement/diagnostics/caliper_benchmark.png",
    output_json_path: str = "measurement/benchmark_results.json",
    num_instances: int = 12,
) -> Dict:
    """Runs automated caliper accuracy benchmarking on held-out test instances."""
    print("=" * 72)
    print("        Measure-It: Digital Caliper Accuracy Benchmark (MO-1)      ")
    print("=" * 72)

    # 1. Select 10+ evaluation instances (isolated test split + late validation split)
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_samples = [line.strip() for line in f if line.strip()]

    with open(val_split_path, "r", encoding="utf-8") as f:
        val_samples = [line.strip() for line in f if line.strip()]

    # Guarantee at least 10+ unseen instances
    eval_instances = test_samples + val_samples[-(num_instances - len(test_samples)) :]
    eval_instances = eval_instances[:num_instances]

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # 2. Initialize Metrology Pipeline
    pipeline = MetrologyPipeline()

    rows: List[Dict] = []
    abs_errors_w: List[float] = []
    abs_errors_h: List[float] = []
    pct_errors_w: List[float] = []
    pct_errors_h: List[float] = []

    print(
        f"{'Instance':<14} | {'GT (mm)':<13} | {'Measured (mm)':<15} | "
        f"{'Err W (mm)':<10} | {'Err H (mm)':<10} | {'MPE (%)':<8} | {'Status'}"
    )
    print("-" * 88)

    for inst_name in eval_instances:
        img_path = Path("dataset/images") / inst_name
        gt = annotations[inst_name]
        gt_w, gt_h = gt["object_nominal_mm"]

        result = pipeline.process_image(img_path, apply_undistort=True)

        if result.status != "SUCCESS":
            print(f"{inst_name:<14} | REJECTED: {result.status}")
            continue

        meas_w = result.width_mm
        meas_h = result.height_mm

        err_w = abs(meas_w - gt_w)
        err_h = abs(meas_h - gt_h)
        pe_w = (err_w / gt_w) * 100.0
        pe_h = (err_h / gt_h) * 100.0
        mean_pe = (pe_w + pe_h) / 2.0

        abs_errors_w.append(err_w)
        abs_errors_h.append(err_h)
        pct_errors_w.append(pe_w)
        pct_errors_h.append(pe_h)

        row_info = {
            "image": inst_name,
            "gt_width_mm": gt_w,
            "gt_height_mm": gt_h,
            "meas_width_mm": round(meas_w, 2),
            "meas_height_mm": round(meas_h, 2),
            "error_width_mm": round(err_w, 3),
            "error_height_mm": round(err_h, 3),
            "abs_error_mean_mm": round((err_w + err_h) / 2.0, 3),
            "pe_width_pct": round(pe_w, 2),
            "pe_height_pct": round(pe_h, 2),
            "mean_pe_pct": round(mean_pe, 2),
            "confidence": round(result.confidence, 3),
            "ppm": round(result.pixels_per_mm, 2),
        }
        rows.append(row_info)

        gt_str = f"{gt_w:.1f} x {gt_h:.1f}"
        meas_str = f"{meas_w:.2f} x {meas_h:.2f}"
        print(
            f"{inst_name:<14} | {gt_str:<13} | {meas_str:<15} | "
            f"{err_w:>8.2f} mm | {err_h:>8.2f} mm | {mean_pe:>7.2f}% | SUCCESS"
        )

    # 3. Compute Aggregate Metrics
    all_abs_errors = abs_errors_w + abs_errors_h
    all_pct_errors = pct_errors_w + pct_errors_h

    mae_total = float(np.mean(all_abs_errors))
    mpe_total = float(np.mean(all_pct_errors))
    max_error = float(np.max(all_abs_errors))
    std_error = float(np.std(all_abs_errors))

    mo1_mae_met = mae_total <= 1.50
    mo1_mpe_met = mpe_total <= 2.00

    print("=" * 88)
    print(f"Total Evaluated Instances : {len(rows)}")
    print(f"Mean Absolute Error (MAE) : {mae_total:.3f} mm  (Threshold: <= 1.50 mm -> {'MET' if mo1_mae_met else 'MISSED'})")
    print(f"Mean Percentage Error(MPE): {mpe_total:.2f}%   (Threshold: <= 2.00%  -> {'MET' if mo1_mpe_met else 'MISSED'})")
    print(f"Standard Deviation of Err : {std_error:.3f} mm")
    print(f"Worst-Case Deviation      : {max_error:.3f} mm")
    print("=" * 88)

    # 4. Save Plot
    Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    x_indices = np.arange(len(rows))
    bar_w = 0.35

    # Absolute Error Plot
    w_errs = [r["error_width_mm"] for r in rows]
    h_errs = [r["error_height_mm"] for r in rows]
    ax1.bar(x_indices - bar_w / 2, w_errs, bar_w, label="Width Error (mm)", color="#2563eb", alpha=0.85)
    ax1.bar(x_indices + bar_w / 2, h_errs, bar_w, label="Height Error (mm)", color="#059669", alpha=0.85)
    ax1.axhline(1.50, color="#dc2626", linestyle="--", linewidth=1.8, label="MO-1 Target Limit (1.50 mm)")
    ax1.axhline(mae_total, color="#1e293b", linestyle=":", linewidth=1.5, label=f"Mean MAE ({mae_total:.2f} mm)")
    ax1.set_title("Per-Instance Absolute Dimension Errors vs Digital Caliper", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Held-out Specimen Instance", fontsize=10)
    ax1.set_ylabel("Absolute Error (mm)", fontsize=10)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([r["image"].replace(".jpg", "") for r in rows], rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, max(2.5, max_error + 0.5))
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Percentage Error Plot
    mean_pes = [r["mean_pe_pct"] for r in rows]
    ax2.plot(x_indices, mean_pes, marker="o", color="#7c3aed", linewidth=2.0, markersize=6, label="Instance MPE (%)")
    ax2.axhline(2.00, color="#dc2626", linestyle="--", linewidth=1.8, label="MO-1 Target Limit (2.0%)")
    ax2.axhline(mpe_total, color="#1e293b", linestyle=":", linewidth=1.5, label=f"Mean MPE ({mpe_total:.2f}%)")
    ax2.set_title("Mean Percentage Error (MPE) Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Held-out Specimen Instance", fontsize=10)
    ax2.set_ylabel("Percentage Error (%)", fontsize=10)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([r["image"].replace(".jpg", "") for r in rows], rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, max(3.5, max([r['mean_pe_pct'] for r in rows]) + 0.8))
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    print(f"[✓] Saved Caliper Benchmark plot to: {output_plot_path}")

    # 5. Save structured JSON
    benchmark_payload = {
        "status": "PASS" if (mo1_mae_met and mo1_mpe_met) else "FAIL",
        "num_instances": len(rows),
        "mae_total_mm": round(mae_total, 4),
        "mpe_total_pct": round(mpe_total, 3),
        "std_error_mm": round(std_error, 4),
        "max_error_mm": round(max_error, 4),
        "mo1_mae_threshold_mm": 1.50,
        "mo1_mpe_threshold_pct": 2.00,
        "mo1_passed": mo1_mae_met and mo1_mpe_met,
        "instances": rows,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"[✓] Saved Benchmark JSON to: {output_json_path}")

    return benchmark_payload


if __name__ == "__main__":
    run_caliper_benchmark()
