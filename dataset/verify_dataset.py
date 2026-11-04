"""Dataset verification, split integrity audit, and diagnostic visualizer."""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset.config import DatasetConfig


def verify_split_files(config: DatasetConfig) -> Dict[str, List[str]]:
    """Audits train, val, test split file lists."""
    splits = {}
    for split_name, expected_count in [
        ("train", config.train_count),
        ("val", config.val_count),
        ("test", config.test_count),
    ]:
        split_path = config.splits_dir / f"{split_name}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing mandatory split file: {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) != expected_count:
            raise ValueError(
                f"Split count mismatch in {split_path.name}: got {len(lines)}, expected {expected_count}"
            )
        splits[split_name] = lines

    # Verify mutual exclusivity (zero test/val leakage into train)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Data leakage detected! Overlaps: Train-Val: {overlap_train_val}, Train-Test: {overlap_train_test}"
        )

    return splits


def verify_dataset_integrity(config: DatasetConfig = None) -> Dict:
    """Audits all images, binary masks, and split consistency."""
    cfg = config or DatasetConfig()
    cfg.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("           Measure-It: Dataset Integrity & Audit Report           ")
    print("=" * 72)

    # 1. Verify Split Files
    splits = verify_split_files(cfg)
    all_files = splits["train"] + splits["val"] + splits["test"]
    print(f"[*] Verified Split Integrity: {len(splits['train'])} Train, {len(splits['val'])} Val, {len(splits['test'])} Test")
    print(f"[*] Total dataset size: {len(all_files)} samples (Zero split leakage)")

    # 2. Verify Image & Mask Pairs
    missing_images = []
    missing_masks = []
    corrupt_masks = []
    areas = []
    aspect_ratios = []

    for fname in all_files:
        base = Path(fname).stem
        img_p = cfg.images_dir / f"{base}.jpg"
        mask_p = cfg.labels_dir / f"{base}.png"

        if not img_p.exists():
            missing_images.append(str(img_p))
            continue
        if not mask_p.exists():
            missing_masks.append(str(mask_p))
            continue

        img = cv2.imread(str(img_p))
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            corrupt_masks.append(fname)
            continue

        # Check resolution matches
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Shape mismatch for {fname}: img {img.shape[:2]} vs mask {mask.shape[:2]}")

        # Check binary values: must only be {0, 255}
        unique_vals = set(np.unique(mask))
        if not unique_vals.issubset({0, 255}):
            corrupt_masks.append(f"{fname} (invalid pixel values: {unique_vals})")
            continue

        # Compute foreground metrics
        fg_pixels = np.sum(mask == 255)
        if fg_pixels < 500:
            corrupt_masks.append(f"{fname} (empty/too small mask: {fg_pixels} px)")
            continue

        areas.append(fg_pixels)

        # Extract contour for oriented bbox
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            w_box, h_box = rect[1]
            if w_box > 0 and h_box > 0:
                aspect = max(w_box, h_box) / min(w_box, h_box)
                aspect_ratios.append(aspect)

    if missing_images or missing_masks or corrupt_masks:
        print(f"[!] Error: Missing Images: {len(missing_images)}, Missing Masks: {len(missing_masks)}, Corrupt: {len(corrupt_masks)}")
        raise RuntimeError("Dataset verification failed.")

    # 3. Generate Diagnostic Visual Overlay
    sample_name = splits["test"][0]
    sample_base = Path(sample_name).stem
    sample_img = cv2.imread(str(cfg.images_dir / f"{sample_base}.jpg"))
    sample_mask = cv2.imread(str(cfg.labels_dir / f"{sample_base}.png"), cv2.IMREAD_GRAYSCALE)

    # Colorize mask contour overlay
    overlay = sample_img.copy()
    contours, _ = cv2.findContours(sample_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)  # Bright green outline

    # Detect ArUco marker in the sample image for visual proof (modern & legacy OpenCV compatible)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(sample_img)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(sample_img, aruco_dict)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids, borderColor=(0, 0, 255))

    diag_overlay_path = cfg.diagnostics_dir / "sample_inspection_overlay.png"
    cv2.imwrite(str(diag_overlay_path), overlay)

    # 4. Generate Dataset Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(areas, bins=16, color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[0].set_title("Object Mask Foreground Pixel Area Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Foreground Pixels", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    axes[1].hist(aspect_ratios, bins=14, color="#1f77b4", edgecolor="black", alpha=0.8)
    axes[1].axvline(
        cfg.object_nominal_height_mm / cfg.object_nominal_width_mm,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Nominal Aspect ({cfg.object_nominal_height_mm / cfg.object_nominal_width_mm:.3f})",
    )
    axes[1].set_title("Object Aspect Ratio Distribution (minAreaRect)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Aspect Ratio (Length / Width)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    dist_plot_path = cfg.diagnostics_dir / "dataset_distribution.png"
    plt.tight_layout()
    plt.savefig(str(dist_plot_path), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 72)
    print("                     DATASET AUDIT SUMMARY                     ")
    print("=" * 72)
    print(f"  Total Valid Pairs        : {len(all_files)}")
    print(f"  Training Set             : {len(splits['train'])} samples (70.0%)")
    print(f"  Validation Set           : {len(splits['val'])} samples (20.0%)")
    print(f"  Testing Set (Isolated)   : {len(splits['test'])} samples (10.0%)")
    print(f"  Mean Object Area         : {np.mean(areas):.0f} px (std: {np.std(areas):.0f})")
    print(f"  Mean BBox Aspect Ratio   : {np.mean(aspect_ratios):.3f} (target: {cfg.object_nominal_height_mm / cfg.object_nominal_width_mm:.3f})")
    print(f"  Mask Value Verification  : PASS (Only {{0, 255}} present)")
    print("-" * 72)
    print(f"[+] Diagnostic overlay saved to: {diag_overlay_path}")
    print(f"[+] Distribution plot saved to : {dist_plot_path}")
    print("[✓] Step 2 (Dataset Acquisition & Annotation) verified.")
    print("=" * 72)

    return {
        "total_samples": len(all_files),
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "mean_area_px": float(np.mean(areas)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "diagnostics_overlay": str(diag_overlay_path),
        "distribution_plot": str(dist_plot_path),
    }


if __name__ == "__main__":
    verify_dataset_integrity()
