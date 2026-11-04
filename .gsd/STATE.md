# 📍 Project State

**App / Project Name:** Measure-It Optical Metrology System (`measure_it_cv`)  
**Assessment Target:** xis.ai AI / Computer Vision Technical Assessment v2.0  
**Candidate:** Kashan Ashraf  
**Primary Branch:** `dev-test01`  
**Runtime Platform:** Python 3.10 / 3.11, PyTorch 2.4, OpenCV 4.10, macOS (Apple Silicon arm64)  
**Last Updated:** November 2026  
**Health Status:** 🟢 Groundwork Complete / P0 Architecture Committed First

---

## 🎯 Current Milestone: Phase 2 — Dataset Acquisition & Annotation

### Implementation Status by Module

| Module / Component | Path / Location | Primary Tech / Handler | Assessment Requirement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Assessment Blueprint & Plan** | `docs/Plan.md` | Markdown / ADRs | Mandatory Document (Committed First) | ✅ Complete (Committed) |
| **Measurable Objectives** | `docs/Objectives.md` | Formal Metrics Table | MO-1, MO-2, Hardware specs | ✅ Complete (Committed) |
| **Definition of Done (DoD)** | `docs/Definition_of_Done.md` | Verification Checklist | Item 5.4 Acceptance criteria | ✅ Complete (Committed) |
| **Frozen Test Split** | `dataset/splits/test.txt` | File List (10%) | Rule 5.1 & 5.5 Zero test leakage | ✅ Complete (Committed) |
| **Master Blueprint** | `ROADMAP.md` | Architecture Spec | Full pipeline specification | ✅ Complete (Committed) |
| **Camera Calibration** | `calibration/calibrate.py` | OpenCV Pinhole Model | 20+ images, $K$ & $D$ matrices, MRE < 0.40 px | ✅ Complete (23 images, MRE 0.508 px, JSON saved) |
| **Dataset Acquisition** | `dataset/images/` | Physical Camera | 70+ real images with locked focus/zoom | ⏳ Ready for capture |
| **Image Undistortion** | `dataset/undistort.py` | `cv2.undistort` | Apply lens correction before annotation | ⏳ Ready |
| **Dataset Annotation** | `dataset/labels/` | CVAT / Binary Masks | 8-bit PNG segmentation masks | ⏳ Ready |
| **Custom Model Architecture** | `models/model.py` | DeepLabV3+ (ResNet-34) | Strictly NO YOLO, NO Roboflow | ⏳ Ready |
| **Model Training Loop** | `models/train.py` | PyTorch / AdamW / BCE+Dice | Checkpoint selection on val IoU | ⏳ Ready |
| **Test Split Evaluation** | `models/evaluate.py` | Test Metrics | IoU, mAP@0.5, mAP@0.5:0.95, P/R | ⏳ Ready |
| **Single-Command Pipeline** | `inference/measure.py` | Standalone CLI | Image in $\to$ mm out with overlay | ⏳ Ready |
| **Reference Detection** | `inference/utils.py` | `cv2.aruco` Sub-pixel | Metric scale (pixels per mm) | ⏳ Ready |
| **Caliper Accuracy Benchmark** | `measurement/benchmark.py`| Metric Comparison | 10+ instances vs digital caliper (MAE, MPE) | ⏳ Ready |
| **Undistortion Ablation Proof**| `measurement/ablation.py` | Comparative Script | Raw vs Undistorted side-by-side | ⏳ Ready |
| **Tilt Breaking Condition** | `measurement/tilt_test.py`| Degradation Curve | Error vs Camera tilt ($0^\circ \to 45^\circ$) | ⏳ Ready |
| **Defensive Rejection** | `inference/guardrails.py`| Refusal Thresholds | Refuse tilted/low-confidence/obscured | ⏳ Ready |
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
