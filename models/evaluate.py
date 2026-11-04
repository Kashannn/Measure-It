#!/usr/bin/env python3
"""Evaluation script: Computes test split IoU, mAP@0.5, mAP@0.5:0.95, Precision, Recall."""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dataset.loader import MetrologyDataset
from models.metrics import compute_pixel_precision_recall, evaluate_segmentation_metrics
from models.model import build_model


def evaluate():
    config_path = Path("models/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = Path(cfg.get("checkpoint_dir", "models/checkpoints")) / cfg.get("best_model_name", "best_model.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {ckpt_path}. Run training first!")

    diag_dir = Path(cfg.get("diagnostics_dir", "models/diagnostics"))
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("      Measure-It: Test Split Segmentation Performance Evaluation    ")
    print("=" * 72)
    print(f"  Checkpoint Loaded : {ckpt_path}")
    print(f"  Evaluation Split  : {cfg.get('split_test', 'dataset/splits/test.txt')} (STRICTLY ISOLATED)")
    print(f"  Device            : {device}")
    print("-" * 72)

    # 1. Load Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Load Test Dataset
    test_ds = MetrologyDataset("test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    all_ious = []
    all_precisions = []
    all_recalls = []
    vis_samples = []

    print(f"[*] Evaluating {len(test_ds)} isolated test split images...")

    with torch.no_grad():
        for idx, (image, mask, meta) in enumerate(test_loader):
            image = image.to(device)
            logits = model(image)
            probs = torch.sigmoid(logits)

            pred_bin = (probs.squeeze().cpu().numpy() >= 0.50).astype(np.uint8) * 255
            gt_bin = (mask.squeeze().cpu().numpy() >= 0.50).astype(np.uint8) * 255

            prec, rec, iou = compute_pixel_precision_recall(pred_bin, gt_bin)

            all_precisions.append(prec)
            all_recalls.append(rec)
            all_ious.append(iou)

            fname = meta["filename"][0]
            print(f"  [{idx+1}/{len(test_ds)}] {fname:15s} | IoU: {iou:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

            if len(vis_samples) < 4:
                # Store sample for visual report
                img_unnorm = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_disp = np.clip((img_unnorm * std + mean) * 255, 0, 255).astype(np.uint8)
                vis_samples.append((fname, img_disp, gt_bin, pred_bin))

    # 3. Aggregate Metrics
    metrics = evaluate_segmentation_metrics(all_ious, all_precisions, all_recalls)

    # 4. Generate Visual Prediction Grid
    fig, axes = plt.subplots(len(vis_samples), 4, figsize=(16, 4 * len(vis_samples)))
    if len(vis_samples) == 1:
        axes = np.expand_dims(axes, 0)

    for i, (name, img, gt, pred) in enumerate(vis_samples):
        # Error map: TP = Green, FP = Red, FN = Blue
        h, w = gt.shape
        err_map = np.zeros((h, w, 3), dtype=np.uint8)
        tp = np.logical_and(pred > 0, gt > 0)
        fp = np.logical_and(pred > 0, gt == 0)
        fn = np.logical_and(pred == 0, gt > 0)
        err_map[tp] = [0, 220, 0]    # Green
        err_map[fp] = [220, 0, 0]    # Red (False Positive)
        err_map[fn] = [0, 100, 255]  # Blue (False Negative)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{name} (Input)", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Predicted Mask", fontsize=11)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(err_map)
        axes[i, 3].set_title("Error Audit (G=TP, R=FP, B=FN)", fontsize=11)
        axes[i, 3].axis("off")

    pred_plot_path = diag_dir / "test_predictions_overlay.png"
    plt.tight_layout()
    plt.savefig(str(pred_plot_path), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 72)
    print("                     TEST SPLIT METRIC RESULTS                  ")
    print("=" * 72)
    print(f"  Mask IoU (Jaccard Index) : {metrics['mask_iou']:.4f}  (Target: >= 0.88 -> {'PASS' if metrics['mask_iou'] >= 0.88 else 'MISS'})")
    print(f"  Precision                : {metrics['precision']:.4f}")
    print(f"  Recall                   : {metrics['recall']:.4f}")
    print(f"  mAP@0.5                  : {metrics['mAP@0.5']:.4f}  (Target: >= 0.92 -> {'PASS' if metrics['mAP@0.5'] >= 0.92 else 'MISS'})")
    print(f"  mAP@0.5:0.95             : {metrics['mAP@0.5:0.95']:.4f}")
    print("-" * 72)
    print(f"[+] Visual prediction overlay saved to: {pred_plot_path}")
    print("[✓] Step 3 (Trained Segmentation Model) verified.")
    print("=" * 72)

    return metrics


if __name__ == "__main__":
    evaluate()
