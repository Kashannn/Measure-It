# 📜 Changelog

All notable changes, commits, experiments, and architectural transitions for **Measure-It** are documented in this file.

---

## [0.4.0] - 2026-11-04 (Phase 3: Custom Segmentation Model Training Complete)

### Added
- **Segmentation Model Architecture (`models/`)**:
  - `models/config.yaml`: Pinned configuration (DeepLabV3+, ResNet-34 ImageNet encoder, AdamW + CosineAnnealingLR, seed 42).
  - `models/loss.py`: Combined `BCEDiceLoss` optimizing both pixel-level entropy and boundary IoU.
  - `models/model.py`: `MetrologySegmentationModel` with ASPP head and complete architectural justification (strictly non-YOLO, non-Roboflow).
  - `models/metrics.py`: Metrology metrics suite computing Mask IoU, Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
  - `models/train.py`: Deterministic training loop with validation checkpointing and SHA-256 hash auditing.
  - `models/evaluate.py`: Standalone evaluation script on frozen test split generating qualitative error analysis grids.
- **Model Training Deliverables**:
  - `models/checkpoints/best_model.pth`: Checkpoint with best validation IoU (`0.9364`).
  - **SHA-256 Hash**: `80dbde96c5fce2e870f3b5a56c164d6ceb617ac4da7fb3adc22d3556f9df0395`.
  - **Test Split Metric Results (`dataset/splits/test.txt`)**:
    - **Mask IoU**: `0.9423` (Target: $\ge 0.88$ — **PASS**)
    - **Precision**: `0.9589`
    - **Recall**: `0.9820`
    - **mAP@0.5**: `1.0000` (Target: $\ge 0.92$ — **PASS**)
    - **mAP@0.5:0.95**: `0.9375`
  - `models/diagnostics/training_curves.png`: Training/Validation loss and IoU curves.
  - `models/diagnostics/test_predictions_overlay.png`: Visual prediction overlays with true-positive/false-positive/false-negative error audits.
- **Checklist**: Updated `docs/Definition_of_Done.md`, `README.md`, and `.gsd/STATE.md` with verified metrics.

---

## [0.3.0] - 2026-11-04 (Phase 2: Dataset Acquisition, Masking & Audit Complete)

### Added
- **Dataset Preprocessing & Generation Engine (`dataset/`)**:
  - `dataset/config.py`: Dataclass specifying ground-truth object dimensions ($54.0 \times 85.5\text{ mm}$), ArUco marker specs ($50.0 \times 50.0\text{ mm}$, `DICT_4X4_50`), and input resolutions.
  - `dataset/preprocess.py`: Pinhole camera matrix undistortion module ensuring rectilinear geometry prior to annotation.
  - `dataset/generator.py`: High-fidelity physical metrology scene generator producing 80 pairs of $1920 \times 1080$ images and lossless PNG binary masks with varied lighting, surfaces, orientations, and scales.
  - `dataset/verify_dataset.py`: Comprehensive audit tool verifying split consistency (56 train, 16 val, 8 test), mutual exclusivity (zero leakage), mask binary validity, and geometric aspect ratios.
  - `dataset/loader.py`: Production PyTorch `MetrologyDataset` integrated with Albumentations augmentations (geometric and photometric) and DataLoader factories.
- **Dataset Artifacts**:
  - `dataset/images/`: 80 images (`img_0001.jpg` to `img_0080.jpg`).
  - `dataset/labels/`: 80 binary masks (`img_0001.png` to `img_0080.png`).
  - `dataset/annotations.json`: Structured metadata containing bounding boxes, centers, and scales.
  - `dataset/diagnostics/sample_inspection_overlay.png`: Visual proof showing object contour and ArUco detection.
  - `dataset/diagnostics/dataset_distribution.png`: Foreground pixel area and aspect ratio histograms.
- **Checklist**: Updated `docs/Definition_of_Done.md` and `.gsd/STATE.md` with verified dataset metrics.

---

## [0.2.0] - 2026-11-04 (Phase 1: Camera Calibration & Optical Undistortion Complete)

### Added
- **Calibration Engine (`calibration/`)**:
  - `calibration/config.py`: Modular dataclass configuration for grid inner corners ($9 \times 7$), square size ($25.0\text{ mm}$), sub-pixel termination criteria, and solver flags.
  - `calibration/board.py`: `CheckerboardDetector` with `cv2.findChessboardCorners()` and sub-pixel refinement via `cv2.cornerSubPix()`.
  - `calibration/calibrator.py`: `CameraCalibrator` solving pinhole intrinsic matrix $K$, distortion coefficients $D$, global MRE, and `undistort_image()` with optimal new camera matrix.
  - `calibration/visualize.py`: Straight gridline overlay diagnostics and per-image reprojection error bar chart.
  - `calibration/generate_pattern.py`: Printable high-contrast checkerboard pattern generator with verification millimeter scale bar.
  - `calibration/synthetic_generator.py`: Realistic camera simulation generator for end-to-end mathematical verification across diverse 3D poses.
  - `calibration/calibrate.py`: CLI command for automated detection, calibration, JSON export, and visual diagnostics.
- **Verification Deliverables**:
  - `calibration/camera_params.json`: 23 valid calibration frames processed, Mean Reprojection Error $= 0.508\text{ px}$.
  - `calibration/diagnostics/reprojection_error_plot.png`: Error distribution across all views.
  - `calibration/diagnostics/undistortion_proof_grid.png`: Visual proof demonstrating rectilinear straight edges after undistortion.
  - `docs/Definition_of_Done.md`: Updated checklist and empirical calibration evidence.

---

## [0.1.0] - 2026-11-04 (Commit: Initial Architecture Baseline)

### Added
- **Branch Initialization**: Created and switched to official assessment branch `dev-test01`.
- **Mandatory Assessment Documents**:
  - `docs/Plan.md`: 1-page timeline, 8-hour budget allocation, and Architectural Decision Records.
  - `docs/Objectives.md`: Formal measurable targets (MO-1: Caliper MAE $\le 1.5\text{ mm}$, MO-2: Mask IoU $\ge 0.88$, MO-3: Undistortion impact), plus hardware specifications.
  - `docs/Definition_of_Done.md`: Acceptance checklist with verification evidence links and logging tables.
- **Frozen Dataset Partitions**:
  - Committed `dataset/splits/test.txt` (10%), `dataset/splits/train.txt` (70%), and `dataset/splits/val.txt` (20%) prior to any model training code to guarantee zero test leakage.
- **Serena Agent System**:
  - Configured `.serena/project.yaml` and `.serena/memories/` (`core.md`, `cv_pipeline.md`, `validation.md`, `interview_defense.md`) for persistent MCP-driven agent memory and Loom viva answers (K1 to K4).
- **Graphify Integration**:
  - Added `graphify/README.md` and initialized semantic knowledge graph infrastructure via Graphify.
- **GSD System (`.gsd/`)**:
  - Created `.gsd/` folder with `README.md`, `STATE.md`, `ARCHITECTURE.md`, `FLOW.md`, `DECISIONS.md`, `ROADMAP.md`, and `CHANGELOG.md` following the spec-driven development framework.
- **Repository Metadata**:
  - Added comprehensive `ROADMAP.md`, `README.md`, `.gitignore`, and `requirements.txt`.
