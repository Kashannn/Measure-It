# 🏛️ System Architecture

**Project:** Measure-It — Optical Metrology & Segmentation System  
**Version:** 2.0  
**Framework:** PyTorch & OpenCV  

---

## 1. Architectural Philosophy & Constraints

Industrial optical metrology requires reproducible physics, rigorous lens geometry, and defensive computer vision. A wrong answer delivered with high confidence is unacceptable.

```text
       ┌────────────────────────────────────────────────────────┐
       │                   Raw Input Photo                      │
       └──────────────────────────┬─────────────────────────────┘
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │     Pinhole Camera Optical Undistortion Engine         │
       │    (cv2.undistort via Calibrated K Matrix & D Coeffs)  │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                   Undistorted Rectilinear Image
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│     DeepLabV3+ Neural Network     │   │      ArUco Fiducial Detector      │
│  (ResNet-34 Backbone + ASPP Head) │   │ (DICT_4X4_50, Sub-Pixel Corners)  │
└─────────────────┬─────────────────┘   └─────────────────┬─────────────────┘
                  ▼                                       ▼
       [ Binary Object Mask ]                 [ Scale: Pixels per mm ]
                  └───────────────────────┬───────────────────────┘
                                          ▼
       ┌────────────────────────────────────────────────────────┐
       │           Geometric Dimension Extractor                │
       │   - Contour Extraction & cv2.minAreaRect Calculation   │
       │   - Aspect Ratio Verification & Tilt Estimation        │
       └──────────────────────────┬─────────────────────────────┘
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │           Defensive Guardrail Filter                   │
       │   - Is Mask Sigmoid Confidence >= 0.85?                │
       │   - Is Plane Tilt <= 20 degrees?                       │
       │   - Is Reference Marker 100% Unobscured?               │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
               PASS                                FAIL
                │                                   │
                ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ Valid Metric Output (mm)      │   │ Defensive Rejection           │
│ - Width:  xx.x mm             │   │ - Status: REFUSED             │
│ - Height: yy.y mm             │   │ - Reason: Non-coplanar tilt   │
│ - Visual Annotated Overlay    │   │           or Low Confidence   │
└───────────────────────────────┘   └───────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Optical Calibration Engine (`calibration/`)
- **Pinhole Projection Model**:
  $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} R & t \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
- **Distortion Modeling**: Corrects 3 radial parameters ($k_1, k_2, k_3$) and 2 tangential parameters ($p_1, p_2$).
- **Optimization**: Levenberg-Marquardt non-linear optimization via `cv2.calibrateCamera`.
- **Mean Reprojection Error (MRE)**:
  $$\text{MRE} = \frac{1}{N} \sum_{i=1}^N \sqrt{\frac{1}{M} \sum_{j=1}^M \| p_{ij} - \hat{p}_{ij} \|^2} < 0.40\text{ px}$$

### 2.2 Neural Segmentation Engine (`models/`)
- **Architecture**: DeepLabV3+ implemented in native PyTorch (`segmentation_models_pytorch`).
- **Encoder**: ResNet-34 pretrained on ImageNet (frozen early layers, fine-tuned high-level feature extractors).
- **Decoder**: Atrous Spatial Pyramid Pooling (ASPP) with dilation rates $[12, 24, 36]$ to preserve sharp boundary contours.
- **Strict Compliance**: Zero dependencies on Ultralytics or Roboflow models.
- **Combined Loss Function**:
  $$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} = - \sum [y \log \hat{y} + (1-y)\log(1-\hat{y})] + \left(1 - \frac{2 |Y \cap \hat{Y}| + \epsilon}{|Y| + |\hat{Y}| + \epsilon}\right)$$

### 2.3 Metric Scaling & Geometric Metrology (`inference/`)
- **Reference Fiducial**: $50.0 \times 50.0\text{ mm}$ OpenCV ArUco square (`cv2.aruco.DICT_4X4_50`).
- **Sub-Pixel Precision**: Corner detection refined using `cv2.cornerSubPix` with zero-zone windowing.
- **Spatial Scale Ratio**:
  $$\text{PPM} = \frac{\text{Perimeter}_{\text{ArUco, px}}}{4 \times 50.0\text{ mm}}$$
- **Rotated Bounding Rectangle**: Uses `cv2.minAreaRect(contour)` to compute the minimum bounding box independent of object orientation.

### 2.4 Defensive Failure Guardrails (`inference/guardrails.py`)
- **Condition 1 (Object Not Found)**: Mask foreground pixels $< 500 \implies$ Exit code `OBJECT_NOT_FOUND`.
- **Condition 2 (Reference Not Found)**: ArUco marker not detected or occluded $\implies$ Exit code `REFERENCE_NOT_FOUND`.
- **Condition 3 (Plane Perspective Tilt)**: Ratio of opposite edges of the ArUco marker exceeds $1.08 \implies$ Exit code `TILTED_PLANE_REJECTED`.
- **Condition 4 (Low Confidence Mask)**: Mean sigmoid output on the predicted mask $< 0.85 \implies$ Exit code `UNCERTAIN_MASK_REJECTED`.

---

## 3. Directory Layout

```text
Measure-It/
├── .gsd/                         # GSD State & Architecture Tracking
├── .serena/                      # Serena MCP Persistent Memory System
├── graphify/                     # Graphify Codebase Knowledge Graph
├── calibration/                  # Camera Calibration Scripts & Images
├── dataset/                      # Physical Images, Masks & Frozen Splits
├── models/                       # DeepLabV3+ Training & Evaluation Scripts
├── inference/                    # Image-to-Millimetres CLI Command
├── measurement/                  # Caliper Benchmarking & Ablation Scripts
├── docs/                         # The 3 Mandatory Assessment Documents
│   ├── Plan.md
│   ├── Objectives.md
│   └── Definition_of_Done.md
├── ROADMAP.md                    # Master Technical Blueprint
├── requirements.txt              # Pinned Dependencies
└── README.md                     # System Documentation
```
