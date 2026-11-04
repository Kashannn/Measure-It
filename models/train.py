#!/usr/bin/env python3
"""Training script for DeepLabV3+ metrology segmentation model."""

import hashlib
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from dataset.loader import get_dataloaders
from models.loss import BCEDiceLoss
from models.metrics import compute_batch_iou
from models.model import build_model


def set_seed(seed: int = 42) -> None:
    """Pins random seed for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sha256(file_path: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def train():
    config_path = Path("models/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("random_seed", 42))

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_dir = Path(cfg.get("checkpoint_dir", "models/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("      Measure-It: Custom Segmentation Model Training Pipeline       ")
    print("=" * 72)
    print(f"  Architecture      : {cfg['architecture']} ({cfg['encoder_name']} ImageNet backbone)")
    print(f"  Execution Device  : {device}")
    print(f"  Epochs / Batch    : {cfg['epochs']} epochs, batch size {cfg['batch_size']}")
    print(f"  Learning Rate     : {cfg['learning_rate']} (AdamW + CosineAnnealingLR)")
    print(f"  Target Constraint : Strictly NO Ultralytics / NO Roboflow (Verified)")
    print("-" * 72)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 0)
    )
    print(f"[*] Loaded Data: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test (Frozen)")

    # 2. Build Model
    model = build_model(cfg).to(device)
    criterion = BCEDiceLoss(
        bce_weight=cfg.get("bce_weight", 0.5), dice_weight=cfg.get("dice_weight", 0.5)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=float(cfg.get("scheduler_eta_min", 1e-6)),
    )

    # Tracking histories
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    best_val_iou = 0.0
    best_model_path = ckpt_dir / cfg.get("best_model_name", "best_model.pth")

    print("\n[*] Commencing model training loop...")
    for epoch in range(1, cfg["epochs"] + 1):
        # --- TRAIN STEP ---
        model.train()
        epoch_train_loss = 0.0
        epoch_train_iou = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)
            epoch_train_iou += compute_batch_iou(logits, masks) * images.size(0)

        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_iou = epoch_train_iou / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_ious.append(train_iou)

        # --- VALIDATION STEP ---
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_iou = 0.0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                epoch_val_loss += loss.item() * images.size(0)
                epoch_val_iou += compute_batch_iou(logits, masks) * images.size(0)

        val_loss = epoch_val_loss / len(val_loader.dataset)
        val_iou = epoch_val_iou / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        scheduler.step()

        # Checkpoint if best validation IoU
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                best_model_path,
            )

        print(
            f"  Epoch [{epoch:02d}/{cfg['epochs']:02d}] "
            f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} "
            f"{'[* Best Checkpoint Saved]' if is_best else ''}"
        )

    # 3. Export Training Curves Plot
    plot_path = ckpt_dir / "training_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, cfg["epochs"] + 1), train_losses, label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(range(1, cfg["epochs"] + 1), val_losses, label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Training and Validation Compound Loss (BCE + Dice)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    axes[1].plot(range(1, cfg["epochs"] + 1), train_ious, label="Train IoU", color="#2ca02c", lw=2)
    axes[1].plot(range(1, cfg["epochs"] + 1), val_ious, label="Val IoU", color="#d62728", lw=2)
    axes[1].axhline(0.88, color="purple", linestyle=":", lw=1.5, label="Objective MO-2 Target (0.88)")
    axes[1].set_title("Mask Intersection over Union (IoU)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("IoU Score", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()

    # Compute SHA-256 hash of saved checkpoint
    sha256_hash = compute_sha256(best_model_path)

    print("\n" + "=" * 72)
    print("                     TRAINING COMPLETION REPORT                 ")
    print("=" * 72)
    print(f"  Best Validation IoU  : {best_val_iou:.4f}")
    print(f"  Saved Checkpoint     : {best_model_path}")
    print(f"  Model SHA-256 Hash   : {sha256_hash}")
    print(f"  Training Curves Plot : {plot_path}")
    print("=" * 72)

    return {
        "best_val_iou": float(best_val_iou),
        "checkpoint_path": str(best_model_path),
        "sha256": sha256_hash,
    }


if __name__ == "__main__":
    train()
