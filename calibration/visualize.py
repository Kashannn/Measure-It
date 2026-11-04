"""Visual diagnostic generators for calibration verification."""

from pathlib import Path
from typing import List, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_straight_gridlines(image: np.ndarray, grid_step: int = 80, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    """Overlays perfectly rectilinear gridlines to visually verify curvature vs straightness."""
    overlay = image.copy()
    h, w = overlay.shape[:2]

    # Draw vertical lines
    for x in range(0, w, grid_step):
        cv2.line(overlay, (x, 0), (x, h), color, thickness, cv2.LINE_AA)

    # Draw horizontal lines
    for y in range(0, h, grid_step):
        cv2.line(overlay, (0, y), (w, y), color, thickness, cv2.LINE_AA)

    return overlay


def generate_undistortion_comparison(
    raw_image: np.ndarray,
    undistorted_image: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Optical Undistortion Verification",
) -> None:
    """Creates a high-resolution side-by-side comparison with gridlines."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply reference straight gridlines
    raw_with_grid = draw_straight_gridlines(raw_image)
    undist_with_grid = draw_straight_gridlines(undistorted_image)

    # Convert BGR to RGB for matplotlib
    raw_rgb = cv2.cvtColor(raw_with_grid, cv2.COLOR_BGR2RGB)
    undist_rgb = cv2.cvtColor(undist_with_grid, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Raw Distorted Frame (Barrel/Pincushion Effect)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(undist_rgb)
    axes[1].set_title("Calibrated Undistorted Frame (Rectilinear Proof)", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(target_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_reprojection_errors(
    per_image_errors: List[dict],
    global_mre: float,
    output_path: Union[str, Path],
) -> None:
    """Plots a bar chart showing the reprojection error for each calibration image."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    filenames = [e["filename"] for e in per_image_errors]
    errors = [e["reprojection_error_px"] for e in per_image_errors]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(filenames)), errors, color="#1f77b4", edgecolor="black", alpha=0.85)

    # Highlight images above threshold
    for i, bar in enumerate(bars):
        if errors[i] > 0.50:
            bar.set_color("#d62728")

    plt.axhline(
        global_mre,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Reprojection Error: {global_mre:.4f} px",
    )
    plt.axhline(
        0.50,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="Acceptance Limit: 0.50 px",
    )

    plt.title("Reprojection Error per Calibration View", fontsize=14, fontweight="bold")
    plt.xlabel("Calibration Image Index", fontsize=12)
    plt.ylabel("Root Mean Square Error (pixels)", fontsize=12)
    plt.xticks(range(len(filenames)), [f"#{i+1}" for i in range(len(filenames))], rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(str(target_path), dpi=150, bbox_inches="tight")
    plt.close()
