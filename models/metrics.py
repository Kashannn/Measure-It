"""Evaluation metrics: Mask IoU, Precision, Recall, mAP@0.5, and mAP@0.5:0.95."""

from typing import Dict, List, Tuple
import numpy as np
import torch


def compute_batch_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor, threshold: float = 0.50) -> float:
    """Computes mean Intersection over Union (Jaccard Index) across a batch."""
    preds = (torch.sigmoid(pred_masks) >= threshold).float()
    targets = (gt_masks > 0.5).float()

    intersection = (preds * targets).sum(dim=(-2, -1))
    union = preds.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) - intersection

    # Avoid zero division
    iou = (intersection + 1e-7) / (union + 1e-7)
    return float(iou.mean().item())


def compute_pixel_precision_recall(
    pred_bin: np.ndarray, gt_bin: np.ndarray
) -> Tuple[float, float, float]:
    """Computes Precision, Recall, and IoU for a single instance."""
    p = (pred_bin > 0).astype(bool)
    g = (gt_bin > 0).astype(bool)

    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    return precision, recall, iou


def compute_map_at_thresholds(
    ious: List[float], thresholds: np.ndarray = np.linspace(0.50, 0.95, 10)
) -> Dict[str, float]:
    """Computes mAP@0.5 and mAP@0.5:0.95 across IoU thresholds."""
    ious_arr = np.array(ious)
    map_scores = {}

    for t in thresholds:
        # A test instance is considered a true positive match if its mask IoU >= t
        hits = (ious_arr >= t).astype(float)
        ap_t = float(np.mean(hits))
        map_scores[f"{t:.2f}"] = ap_t

    map_50 = map_scores["0.50"]
    map_50_95 = float(np.mean(list(map_scores.values())))

    return {
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": map_50_95,
        "threshold_scores": map_scores,
    }


def evaluate_segmentation_metrics(
    all_ious: List[float],
    all_precisions: List[float],
    all_recalls: List[float],
) -> Dict[str, float]:
    """Aggregates all evaluation metrics into final report dict."""
    map_dict = compute_map_at_thresholds(all_ious)

    return {
        "mask_iou": float(np.mean(all_ious)),
        "precision": float(np.mean(all_precisions)),
        "recall": float(np.mean(all_recalls)),
        "mAP@0.5": map_dict["mAP@0.5"],
        "mAP@0.5:0.95": map_dict["mAP@0.5:0.95"],
    }
