# Measure-It: Real-World Object Segmentation & Metric Metrology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Branch](https://img.shields.io/badge/branch-dev--test01-blue.svg)](https://github.com/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-brightgreen.svg)](https://www.python.org/)
[![Assessment](https://img.shields.io/badge/xis.ai-CV%20Assessment%20v2.0-red.svg)](docs/Plan.md)

An end-to-end industrial computer vision pipeline that photographs a physical object and calculates its real-world dimensions (width and height in millimeters) alongside an uncertainty confidence metric. The system incorporates camera pinhole calibration, optical lens undistortion, custom semantic segmentation (DeepLabV3+ with ResNet-34 backbone), coplanar metric fiducial detection, and defensive error refusal.

---

## 1. System Architecture & Workflow

```text
       [ Raw Photo ]
             │
             ▼
   [ Lens Undistortion ]  <--- Loaded Intrinsics (K) & Distortion (D)
             │
      ┌──────┴──────────────────────────┐
      ▼                                 ▼
[ DeepLabV3+ SegNet ]         [ ArUco Marker Detector ]
  (Object Mask Contour)         (Scale: Pixels per Millimetre)
      └──────┬──────────────────────────┘
             ▼
   [ Perspective / BBox ]     <--- minAreaRect Dimension Fitting
             │
             ▼
  [ Defensive Guardrails ]    <--- Rejects Tilt > 20°, Low Conf < 0.85
             │
             ▼
   [ Metric Output (mm) ]     <--- Width (mm), Height (mm), Overlay Image
```

---

## 2. Hardware Profile (Assessment Testing Platform)

All reported inference latencies, calibration errors, and accuracy metrics are produced directly on this host environment:

| Specification | Hardware Value |
|---|---|
| **OS** | macOS Sonoma / Darwin (Apple Silicon arm64) |
| **CPU** | Apple M-series Silicon |
| **GPU / Acceleration** | Apple Metal Performance Shaders (MPS) / CUDA compatible |
| **RAM** | 16 GB Unified Memory |
| **Camera** | Smartphone Camera (Locked 1.0x optical zoom, locked focus, locked exposure) |
| **Image Resolution** | $1920 \times 1080$ px |

---

## 3. Dataset Summary

- **Physical Target Object**: Rigid planar object with sharp orthogonal boundaries.
- **Reference Fiducial**: OpenCV ArUco Marker (Dictionary: `DICT_4X4_50`, ID: 0, Physical size: $50.0 \times 50.0\text{ mm}$ verified with digital caliper).
- **Total Images Captured**: 80 physical photos taken across varied backgrounds and lighting.
- **Data Preprocessing**: All images optical undistortion applied prior to labeling.
- **Labeling Tool**: CVAT (Computer Vision Annotation Tool) producing pixel-accurate binary masks.
- **Split Distribution**:
  - **Train Split (70%)**: 56 images (`dataset/splits/train.txt`)
  - **Validation Split (20%)**: 16 images (`dataset/splits/val.txt`)
  - **Test Split (10%)**: 8 images (`dataset/splits/test.txt`) — **Committed prior to model training**.

---

## 4. Setup & Installation

### Environment Setup
```bash
# Clone the repository
git clone <repo_url>
cd Measure-It

# Ensure branch dev-test01
git checkout dev-test01

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install reproducible dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Execution Pipeline (Stage by Stage)

### Stage 1: Camera Calibration
Extracts checkerboard inner corners, computes camera matrix $K$ and distortion coefficients $D$, and reports Mean Reprojection Error (MRE).
```bash
python calibration/calibrate.py --images_dir calibration/images --output calibration/camera_params.json
```

### Stage 2: Train Custom Segmentation Model
Trains DeepLabV3+ with ResNet-34 backbone (strictly avoiding YOLO/Roboflow).
```bash
python models/train.py --config models/config.yaml
```

### Stage 3: Evaluate Model Metrics on Frozen Test Split
Computes test mask IoU, mAP@0.5, mAP@0.5:0.95, Precision, and Recall.
```bash
python models/evaluate.py --weights models/checkpoints/best_model.pth --split dataset/splits/test.txt
```

### Stage 4: Single Command (Image to Millimetres)
Takes a raw photograph, performs lens undistortion, segments the object, detects the reference marker, and outputs dimensions in millimeters.
```bash
python inference/measure.py --image path/to/sample.jpg --output overlay.jpg
```
*Output Format:*
```json
{
  "status": "SUCCESS",
  "width_mm": 54.21,
  "height_mm": 85.58,
  "confidence": 0.942,
  "pixels_per_mm": 11.45,
  "overlay_saved_to": "overlay.jpg"
}
```

### Stage 5: Ground Truth Caliper Benchmarking & Undistortion Proof
```bash
# Caliper comparison over 10+ held-out instances
python measurement/benchmark_accuracy.py

# Comparative ablation: Raw (distorted) vs Undistorted measurement delta
python measurement/ablation_undistort.py
```

---

## 6. Model Weights & Reproducibility

- **Model Architecture**: DeepLabV3+ (Encoder: `resnet34`, ImageNet Pretrained, Decoder: Atrous Spatial Pyramid Pooling)
- **Framework Compliance**: Pure PyTorch (`segmentation_models_pytorch`). Strictly avoids prohibited YOLO / Roboflow frameworks.
- **Model Checkpoint**: Saved locally at `models/checkpoints/best_model.pth`.
- **Checkpoint SHA-256 Hash**:
  ```text
  80dbde96c5fce2e870f3b5a56c164d6ceb617ac4da7fb3adc22d3556f9df0395
  ```
- **Test Set Performance**:
  - **Mask IoU**: `0.9423` (Target $\ge 0.88$ — **PASS**)
  - **Precision**: `0.9589`
  - **Recall**: `0.9820`
  - **mAP@0.5**: `1.0000` (Target $\ge 0.92$ — **PASS**)
  - **mAP@0.5:0.95**: `0.9375`

---

## 7. Assessment Documents

As mandated by Section 5.2 of the technical assessment specification:
- [docs/Plan.md](docs/Plan.md) — 1-Page Time Allocation, Strategy & Architectural Decision Records.
- [docs/Objectives.md](docs/Objectives.md) — Measurable Objectives (MO-1, MO-2, MO-3) and Environmental Specifications.
- [docs/Definition_of_Done.md](docs/Definition_of_Done.md) — Acceptance Criteria and Verification Evidence Checklist.
- [ROADMAP.md](ROADMAP.md) — Master Technical Implementation Blueprint.
