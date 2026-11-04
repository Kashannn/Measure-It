# Project Plan: Optical Object Measurement System

> **Candidate**: Kashan Ashraf  
> **Role Assessment**: AI / Computer Vision at xis.ai  
> **Branch**: `dev-test01`  
> **Allocated Working Time**: 8 Hours (within a 24-Hour submission window)  
> **Document Status**: Committed First (Prior to any training or implementation code)

---

## 1. Time Allocation Breakdown (8 Hours Total)

| Phase | Task Description | Planned Duration | Cumulative Time | Priority |
|---|---|---|---|---|
| **Phase 1** | **Camera Calibration & Undistortion**: Checkerboard capture (25+ images), corner detection, intrinsic matrix $K$, distortion coefficients $D$, reprojection error calculation, and image undistortion validation. | 1.0 hr | 1.0 hr | **P0 (Critical)** |
| **Phase 2** | **Dataset Acquisition & Annotation**: Capture 80 high-resolution images with locked camera settings. Undistort images. Annotate binary segmentation masks (CVAT). Freeze 70/20/10 split. | 1.5 hrs | 2.5 hrs | **P0 (Critical)** |
| **Phase 3** | **Model Architecture & Training Setup**: Implement PyTorch Dataset, Albumentations transforms, DeepLabV3+ with ResNet-34 backbone (No YOLO / Roboflow). Launch training loop with early stopping. *(Training runs independently while next tasks are coded)*. | 1.0 hr | 3.5 hrs | **P0 (Critical)** |
| **Phase 4** | **Inference & Metrology Engine**: Implement `inference/measure.py` CLI command. ArUco reference detection, homography / scale factor (pixels-to-mm), rotated minimum bounding box extraction, confidence thresholding, and diagnostic overlay generation. | 1.5 hrs | 5.0 hrs | **P0 (Critical)** |
| **Phase 5** | **Ground-Truth Caliper Benchmarking & Undistortion Proof**: Benchmark 10+ real physical instances against digital caliper. Calculate MAE and MPE. Generate comparative side-by-side table on raw vs undistorted images to prove undistortion impact. | 1.0 hr | 6.0 hrs | **P0 (Critical)** |
| **Phase 6** | **Failure Boundary & Rejection Guardrails**: Test camera tilt breaking condition ($0^\circ, 15^\circ, 30^\circ, 45^\circ$). Implement refusal logic for low confidence masks and occluded/skewed reference targets. | 1.0 hr | 7.0 hrs | **P1 (High)** |
| **Phase 7** | **Loom Recording & Documentation Finalization**: Record <5:00 min video demonstration answering K1-K4 viva questions on camera. Finalize `Definition_of_Done.md`, SHA-256 model weights, and submit Pull Request. | 1.0 hr | 8.0 hrs | **P0 (Critical)** |

---

## 2. Order of Operations

1. **Commit Zero**: Commit `Plan.md`, `Objectives.md`, `Definition_of_Done.md`, and frozen `dataset/splits/test.txt`.
2. **Step 1 (Calibration)**: Run calibration, save `camera_params.json`, report reprojection error. Commit `calibration/`.
3. **Step 2 (Data Pipeline)**: Store undistorted images, annotation masks, and split file lists. Commit `dataset/`.
4. **Step 3 (Training & Evaluation)**: Train DeepLabV3+, save checkpoint weights, record test split IoU and mAP. Commit `models/`.
5. **Step 4 (Inference Command)**: Build single CLI command taking one image to mm dimensions. Commit `inference/`.
6. **Step 5 (Validation & Experiments)**: Caliper accuracy table, raw vs undistorted comparison, tilt breaking test, rejection thresholds. Commit `measurement/`.
7. **Step 6 (Deliverables & PR)**: Verify all DoD checklist items, record 5-minute Loom video, open PR into `main`.

---

## 3. Deliberately Skipped Items (Time Management)

To ensure completion of the core floor requirements (Items 1 to 6) and defensive guardrails (Items 8 to 10), the following have been explicitly deprioritized:
1. **Interactive Web/GUI Application**: Skipped in favor of a clean, robust CLI script (`python inference/measure.py --image <path>`) as required by Item 4.
2. **Multi-Class Semantic Segmentation**: Skipped because only the target object and reference marker are needed for single-object industrial inspection.
3. **Complex Stereo/Multi-View Depth Estimation**: Monocular single-view metrology with coplanar ArUco reference fulfills all requirements with minimal computational overhead.

---

## 4. Architectural Decision Record (ADR)

* **Decision 1: Segmentation Framework**
  * *Selected*: PyTorch with `segmentation_models_pytorch` using **DeepLabV3+** and a **ResNet-34** backbone.
  * *Rejected*: Ultralytics YOLOv8-seg, Mask R-CNN, and Roboflow Hosted Inference.
  * *Reason & Cost*: Ultralytics and Roboflow are strictly forbidden by assessment rules. Mask R-CNN has significantly heavier inference overhead and slower convergence on small custom datasets. DeepLabV3+ with ASPP (Atrous Spatial Pyramid Pooling) excels at capturing boundary details on geometric contours while training in under 15 minutes on a modern GPU.

* **Decision 2: Metric Reference Selection**
  * *Selected*: OpenCV ArUco Marker (Dictionary: `DICT_4X4_50`, Marker ID 0, $50.0 \times 50.0\text{ mm}$ size).
  * *Rejected*: Standard coin (e.g., 10-cent coin) or credit card edge detection.
  * *Reason & Cost*: ArUco markers provide sub-pixel corner detection (`cv2.cornerSubPix`) and deterministic orientation ID, preventing lighting-induced boundary ambiguities that plague circular coins or worn credit cards.

* **Decision 3: Metric Dimension Calculation Method**
  * *Selected*: Minimum-area rotated bounding rectangle (`cv2.minAreaRect`) applied on the segmented binary mask contour.
  * *Rejected*: Axis-aligned bounding box (`cv2.boundingRect`) or manual extreme points.
  * *Reason & Cost*: Axis-aligned bounding boxes fail as soon as the target object is placed at any rotation angle relative to the image frame, overestimating dimensions by up to 40%. `minAreaRect` calculates true principal dimensions regardless of object placement orientation.
