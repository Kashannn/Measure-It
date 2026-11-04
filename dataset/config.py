"""Configuration parameters for the Metrology Dataset and Preprocessing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Rigid specifications for real-world object dataset."""

    # Physical Ground-Truth Object Dimensions (Millimetres)
    # Standard precision test specimen: 54.00 mm x 85.50 mm (aspect ratio ~ 1.583)
    object_nominal_width_mm: float = 54.00
    object_nominal_height_mm: float = 85.50
    object_name: str = "Precision Rectangular Specimen Block"

    # Physical Reference ArUco Marker Specifications
    aruco_dict_name: str = "DICT_4X4_50"
    aruco_marker_id: int = 0
    aruco_size_mm: float = 50.00  # 50.0 mm x 50.0 mm calibrated square

    # Image Resolutions
    capture_resolution: Tuple[int, int] = (1920, 1080)  # (Width, Height)
    model_input_size: Tuple[int, int] = (512, 512)     # (Width, Height)

    # Dataset Partitioning (70% Train, 20% Val, 10% Test across 80 captures)
    total_images: int = 80
    train_count: int = 56
    val_count: int = 16
    test_count: int = 8

    # Paths
    dataset_root: Path = field(default_factory=lambda: Path("dataset"))
    images_dir: Path = field(default_factory=lambda: Path("dataset/images"))
    labels_dir: Path = field(default_factory=lambda: Path("dataset/labels"))
    splits_dir: Path = field(default_factory=lambda: Path("dataset/splits"))
    diagnostics_dir: Path = field(default_factory=lambda: Path("dataset/diagnostics"))
