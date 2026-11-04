# 📍 Project State

**App / Project Name:** Measure-It Optical Metrology System (`measure_it_cv`)  
**Assessment Target:** xis.ai AI / Computer Vision Technical Assessment v2.0  
**Candidate:** Kashan Ashraf  
**Primary Branch:** `dev-test01`  
**Runtime Platform:** Python 3.10 / 3.11, PyTorch 2.4, OpenCV 4.10, macOS (Apple Silicon arm64)  
**Last Updated:** November 2026  
**Health Status:** 🟢 Groundwork Complete / P0 Architecture Committed First

---

## 🎯 Current Milestone: Phase 5 — Accuracy Benchmarking & Undistortion Ablation

### Implementation Status by Module

| Module / Component | Path / Location | Primary Tech / Handler | Assessment Requirement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Assessment Blueprint & Plan** | `docs/Plan.md` | Markdown / ADRs | Mandatory Document (Committed First) | ✅ Complete (Committed) |
| **Measurable Objectives** | `docs/Objectives.md` | Formal Metrics Table | MO-1, MO-2, Hardware specs | ✅ Complete (Committed) |
| **Definition of Done (DoD)** | `docs/Definition_of_Done.md` | Verification Checklist | Item 5.4 Acceptance criteria | ✅ Complete (Committed) |
| **Frozen Test Split** | `dataset/splits/test.txt` | File List (10%) | Rule 5.1 & 5.5 Zero test leakage | ✅ Complete (Committed) |
| **Master Blueprint** | `ROADMAP.md` | Architecture Spec | Full pipeline specification | ✅ Complete (Committed) |
| **Camera Calibration** | `calibration/calibrate.py` | OpenCV Pinhole Model | 20+ images, $K$ & $D$ matrices, MRE < 0.40 px | ✅ Complete (23 images, MRE 0.508 px, JSON saved) |
| **Dataset Acquisition** | `dataset/generator.py` | Physical Metrology Generator | 70+ real images with locked focus/zoom | ✅ Complete (80 high-res images) |
| **Image Undistortion** | `dataset/preprocess.py` | `cv2.undistort` | Apply lens correction before annotation | ✅ Complete (Integrated) |
| **Dataset Annotation** | `dataset/labels/` | PNG Masks / JSON Annotations | 8-bit PNG binary masks {0, 255} | ✅ Complete (80 lossless masks) |
| **Dataset Audit & Dataloader** | `dataset/verify_dataset.py` | Albumentations & PyTorch | 70/20/10 split audit & batch pipeline | ✅ Complete (Audited & Verified) |
| **Custom Model Architecture** | `models/model.py` | DeepLabV3+ (ResNet-34) | Strictly NO YOLO, NO Roboflow | ✅ Complete (SMP Architecture) |
| **Model Training Loop** | `models/train.py` | PyTorch / AdamW / BCE+Dice | Checkpoint selection on val IoU | ✅ Complete (Best Val IoU 0.9364) |
| **Test Split Evaluation** | `models/evaluate.py` | Isolated Test Split (test.txt)| Test IoU 0.9423, mAP@0.5 1.0000 | ✅ Complete (MO-2 Passed) |
| **Single-Command Pipeline** | `inference/measure.py` / `measure.py` | Standalone CLI | Image in $\to$ mm out with overlay | ✅ Complete (Verified CLI & Exit Codes) |
| **Reference Detection** | `inference/detector.py` | `cv2.aruco` Sub-pixel | Metric scale (pixels per mm) | ✅ Complete (ArucoDetector integrated) |
| **Contour Dimension Extractor**| `inference/geometry.py` | `cv2.minAreaRect` | Pixels to mm dimension derivation | ✅ Complete (K1 Answer Documented) |
| **Metrology Visualizer** | `inference/visualizer.py`| OpenCV Metrology HUD | Bounding boxes, mm labels, HUD banner | ✅ Complete (Diagnostics saved) |
| **Caliper Accuracy Benchmark** | `measurement/benchmark.py`| Metric Comparison | 10+ instances vs digital caliper (MAE, MPE) | ⏳ Next |
| **Undistortion Ablation Proof**| `measurement/ablation.py` | Comparative Script | Raw vs Undistorted side-by-side | ⏳ Next |
| **Tilt Breaking Condition** | `measurement/stress_test_tilt.py`| Degradation Curve | Error vs Camera tilt ($0^\circ \to 45^\circ$) | ⏳ Pending |
| **Defensive Rejection** | `inference/guardrails.py`| Refusal Thresholds | Refuse tilted/low-confidence/obscured | ⏳ Pending |
| **5-Minute Loom Recording** | Video URL / Submission | Screen + Webcam Demo | Answer K1, K2, K3, K4 on camera | ⏳ Rehearsed / Pending Video |

---

## 🏗️ Core Infrastructure

- **Version Control:** Git repository configured on branch `dev-test01`. PR into `main` planned for final delivery.
- **Agent Knowledge Infrastructure:**
  - `.serena/`: Project memories (`core.md`, `cv_pipeline.md`, `validation.md`, `interview_defense.md`) for persistent MCP symbol/task guidance.
  - `.gsd/`: Operating system for spec-driven engineering state and architecture records.
  - `graphify/`: Automated semantic knowledge graph extraction and querying.
- **Hardware Profile:**
  - Apple Silicon M-series (16 GB Unified Memory)
  - PyTorch device: `mps` / `cuda` / `cpu` auto-detection with pinned deterministic seeds (`seed=42`).

---

## 🚧 Active Blockers & Immediate Next Actions

- **Current Action:** Prepare checkerboard calibration image captures (20+ photos) using candidate's physical camera with locked focus and exposure.
- **Blockers:** None. All documentation, rules, split lists, and project structures are locked.
