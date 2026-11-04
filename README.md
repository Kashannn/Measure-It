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
- [docs/LOOM_SCRIPT.md](docs/LOOM_SCRIPT.md) — Timed 5-Minute Video Recording Script & Cue Cards.
- [ROADMAP.md](ROADMAP.md) — Master Technical Implementation Blueprint.

---

## 8. Empirical Benchmark & Validation Results

### A. Caliper Accuracy Benchmark (MO-1 Met)
Benchmarked across 12 held-out specimens ([`measurement/benchmark_accuracy.py`](measurement/benchmark_accuracy.py)):

| Metric | Target Standard | Measured Result | Status |
|---|---|---|:---:|
| **Mean Absolute Error (MAE)** | $\le 1.50\text{ mm}$ | **1.433 mm** | **MET** |
| **Mean Percentage Error (MPE)** | $\le 2.0\%$ | **1.97%** | **MET** |
| **Standard Deviation of Error** | — | **1.311 mm** | — |
| **Worst-Case Deviation** | $\le 6.0\text{ mm}$ | **5.620 mm** | **MET** |

Visual proof: [`measurement/diagnostics/caliper_benchmark.png`](measurement/diagnostics/caliper_benchmark.png)

### B. Undistortion Ablation Proof (Item 6 Met)
Direct comparison of same instances on raw sensor vs calibrated undistorted frames ([`measurement/ablation_undistort.py`](measurement/ablation_undistort.py)):

| Condition | Mean Absolute Error (MAE) | Mean Percentage Error |
|---|---|---|
| **Raw Distorted Sensor Images** | $3.169\text{ mm}$ | $4.36\%$ |
| **Undistorted Calibrated Images** | **$1.433\text{ mm}$** | **$1.97\%$** |
| **Empirical Error Reduction Delta** | **$-1.735\text{ mm}$** | **$54.8\%$ reduction** |

Visual proof: [`measurement/diagnostics/undistortion_ablation.png`](measurement/diagnostics/undistortion_ablation.png)

### C. Tilt Stress Test & Breaking Angle (Item 8 Met)
Quantitative degradation curve under out-of-plane pitch angles ($0^\circ \to 45^\circ$):
- **Operational Zone ($0^\circ \text{--} 15^\circ$)**: MAE $\le 1.50\text{ mm}$, highly reliable.
- **Empirical Breaking Threshold**: Pitch **$\ge 20^\circ$** causes perspective foreshortening failure where measurement error explodes beyond $48\text{ mm}$ ($> 74\%$ MPE).
Visual proof: [`measurement/diagnostics/tilt_degradation_curve.png`](measurement/diagnostics/tilt_degradation_curve.png)

### D. Defensive Rejection Guardrails (Item 9 Met)
Refuses untrustworthy inputs rather than returning corrupt numbers ([`inference/guardrails.py`](inference/guardrails.py)):
- **Exit Code 0 (`SUCCESS`)**: Clean valid measurement computed.
- **Exit Code 1 (`ERROR`)**: Missing or unreadable image file.
- **Exit Code 2 (`OBJECT_NOT_FOUND`)**: Target specimen absent or mask $< 5000\text{ px}$.
- **Exit Code 3 (`REFERENCE_NOT_FOUND`)**: ArUco fiducial missing or distorted.
- **Exit Code 4 (`UNTRUSTWORTHY_INPUT_REJECTED`)**: Refused due to severe tilt ($> 1.10$ skew), blur (Laplacian $< 80$), underexposure ($< 40$), or low model confidence ($< 0.85$).

---

## 9. Viva Defense & Technical Interview (Questions K1 to K4)

| Question | Core Answer Summary | Exact Code Reference |
|---|---|---|
| **K1: Which file & line turns px into mm?** | Pixel dimensions from `cv2.minAreaRect` are divided by `pixels_per_mm`: `width_mm = w_px / PPM` and `height_mm = h_px / PPM`. Requires coplanar reference and specimen. | [`inference/geometry.py#L82-L85`](inference/geometry.py#L82-L85) & [`inference/detector.py#L87-L95`](inference/detector.py#L87-L95) |
| **K2: Where is undistortion applied & impact?** | Applied in `undistort_image()` via `cv2.undistort(K, D)` before segmentation and ArUco detection. Reduces error from $3.17\text{ mm}$ to $1.43\text{ mm}$ (54.8% error reduction). | [`inference/pipeline.py#L100-L105`](inference/pipeline.py#L100-L105) & [`measurement/ablation_undistort.py`](measurement/ablation_undistort.py) |
| **K3: Model architecture & failure modes?** | Custom DeepLabV3+ with ResNet-34 backbone and ASPP head (pure PyTorch, non-YOLO/non-Roboflow). Breaks under steep tilt $\ge 20^\circ$, caught by defensive refusal. | [`models/model.py`](models/model.py) & [`inference/guardrails.py`](inference/guardrails.py) |
| **K4: What to improve & what NOT to do?** | **Improve**: Snakes / GrabCut sub-pixel edge refinement and 4-point planar homography warp. **NOT DO**: Replace segmentation with bounding box object detection (cannot fit rounded corners, encapsulates background clutter). | Full discussion in [`docs/LOOM_SCRIPT.md`](docs/LOOM_SCRIPT.md) |
