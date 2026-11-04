"""PyTorch Dataset, Albumentations transforms, and DataLoader factories."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.config import DatasetConfig


class MetrologyDataset(Dataset):
    """PyTorch Dataset for semantic segmentation of industrial metrology objects."""

    def __init__(
        self,
        split: str = "train",
        config: Optional[DatasetConfig] = None,
        transform: Optional[A.Compose] = None,
    ):
        self.split = split
        self.config = config or DatasetConfig()
        self.split_file = self.config.splits_dir / f"{split}.txt"

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {self.split_file}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.file_names = [line.strip() for line in f if line.strip()]

        self.transform = transform or self.get_default_transforms(split)

    def get_default_transforms(self, split: str) -> A.Compose:
        """Returns standard augmentation pipeline tailored for metrology."""
        w_in, h_in = self.config.model_input_size

        if split == "train":
            return A.Compose(
                [
                    A.Resize(h_in, w_in),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.08,
                        rotate_limit=45,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            # Deterministic resizing and normalization for validation and testing
            return A.Compose(
                [
                    A.Resize(h_in, w_in),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        fname = self.file_names[idx]
        base_name = Path(fname).stem

        img_path = self.config.images_dir / f"{base_name}.jpg"
        mask_path = self.config.labels_dir / f"{base_name}.png"

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Binarize mask to 0.0 or 1.0 float32
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask has channel dimension (1, H, W)
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0)

        meta = {
            "filename": fname,
            "orig_size": [image.shape[-2], image.shape[-1]],
        }

        return image, mask, meta


def get_dataloaders(
    batch_size: int = 8,
    num_workers: int = 2,
    config: Optional[DatasetConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating Train, Validation, and Test PyTorch DataLoaders."""
    cfg = config or DatasetConfig()

    train_ds = MetrologyDataset("train", config=cfg)
    val_ds = MetrologyDataset("val", config=cfg)
    test_ds = MetrologyDataset("test", config=cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Single-image evaluation for metric metrology
        shuffle=False,
        num_workers=1,
    )

    return train_loader, val_loader, test_loader
