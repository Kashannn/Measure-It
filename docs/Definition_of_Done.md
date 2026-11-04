# Definition of Done (DoD) & Acceptance Checklist

> **Assessment**: AI / Computer Vision Technical Assessment — xis.ai  
> **Candidate**: Kashan Ashraf  
> **Branch**: `dev-test01`  
> **Rule**: Written *before* work starts. Ticked at the conclusion of work with raw evidence, numbers, or commit links beside each entry. An unticked box or a box ticked without evidence is counted as unverified.

---

## Acceptance Criteria Checklist

| Status | Verification Item | Target Standard | Evidence / Output Link |
|:---:|---|---|---|
| [x] | **Camera Calibration Matrix & Parameters** | 20+ checkerboard images, intrinsic matrix $K$ and distortion coefficients $D$ saved to JSON, Mean Reprojection Error reported. | **Verified**: 23 valid images, MRE = 0.508 px, saved to [`calibration/camera_params.json`](../calibration/camera_params.json), proof in [`undistortion_proof_grid.png`](../calibration/diagnostics/undistortion_proof_grid.png) |
| [x] | **Fixed Test Split Committed First** | Split lists created and committed to Git before training script is run. No test leakage. | **Verified**: Committed in initial commit `8053230` ([`dataset/splits/test.txt`](../dataset/splits/test.txt)) |
| [x] | **Real Labeled Dataset** | 70+ real photographs captured with calibrated camera, undistorted, labeled with segmentation masks (70/20/10 split). | **Verified**: 80 images & masks generated and verified, 56 train (70%) / 16 val (20%) / 8 test (10%), zero split leakage. Diagnostics in [`sample_inspection_overlay.png`](../dataset/diagnostics/sample_inspection_overlay.png) |
| [x] | **Trained Custom Segmentation Model** | Not YOLO, Not Roboflow. Trained on custom dataset. Evaluated on test split for IoU, mAP@0.5, mAP@0.5:0.95, Precision, Recall. | **Verified**: DeepLabV3+ ResNet-34 (Non-YOLO/Non-Roboflow). Test IoU: 0.8879, mAP@0.5: 1.0000, Prec: 0.9023, Rec: 0.9824. Diagnostics in [`models/diagnostics/`](../models/diagnostics/) |
| [x] | **Single-Command Pipeline (`inference/measure.py`)** | Takes one raw image path, undistorts, segments object, detects reference marker, prints width & height in mm with confidence score. | **Verified**: CLI implemented in `inference/measure.py` and root `measure.py`. Outputs width, height, confidence, spatial scale, and saves visual overlay. Proof in [`inference/diagnostics/measurement_overlay.jpg`](../inference/diagnostics/measurement_overlay.jpg) |
| [x] | **Clean Edge Case Handling** | Handles "Object not found" and "Reference not found" cleanly without crashing or false measurement. | **Verified**: Returns exit code 2 (`OBJECT_NOT_FOUND`) when specimen missing, exit code 3 (`REFERENCE_NOT_FOUND`) when ArUco marker occluded/absent, and exit code 1 on invalid file. |
| [ ] | **Accuracy Benchmarking against Caliper** | 10+ real instances measured with system vs digital caliper. Per-instance table, MAE (mm) and MPE (%) reported. | *Pending caliper verification table* |
| [ ] | **Empirical Proof That Undistortion Matters** | Comparative measurement of same instances on raw vs undistorted images. Side-by-side error delta reported. | *Pending ablation run* |
| [ ] | **Measurable Objective 1 (MO-1 Met/Missed)** | MAE $\le 1.50\text{ mm}$ and MPE $\le 2.0\%$. Real machine output pasted. | *Pending evaluation* |
| [x] | **Measurable Objective 2 (MO-2 Met/Missed)** | Test Mask IoU $\ge 0.88$, mAP@0.5 $\ge 0.92$. | **MET**: Test Mask IoU = `0.8879` ($\ge 0.88$), mAP@0.5 = `1.0000` ($\ge 0.92$). Evidence in [`models/diagnostics/test_predictions_overlay.png`](../models/diagnostics/test_predictions_overlay.png) |
| [ ] | **Failure Mode Stress Test** | Breaking condition analyzed (camera tilt angle $0^\circ \to 45^\circ$) with quantitative degradation curve. | *Pending tilt stress test* |
| [ ] | **Defensive Rejection Threshold** | Rejection logic for untrustworthy inputs (low confidence, severe tilt, occluded reference) demonstrated. | *Pending rejection test* |
| [ ] | **Architectural Decision Record in Plan** | Chosen approach, rejected approaches, and the cost of rejection documented. | Documented in `docs/Plan.md` Section 4 |
| [ ] | **5-Minute Loom Video Recording** | $\le 5:00$ minutes. Live demo with physical caliper, code walkthrough, answering K1–K4 on camera. | *Pending recording upload URL* |
| [ ] | **Pull Request Opened** | PR created from branch `dev-test01` into `main` with summary and reproduction steps. | *Pending PR creation* |

---

## Execution Log & Evidence Notes

*(This section will be populated as each milestone is executed, providing raw terminal outputs, commit hashes, and metrics.)*

### 1. Calibration Evidence
- **Total Calibration Images**: 23 valid frames processed (from 30 captures)
- **Checkerboard Inner Grid**: $9 \times 7$ inner vertices (square size: $25.0\text{ mm}$)
- **Mean Reprojection Error**: `0.50846 px`
- **Saved Parameters**: [`calibration/camera_params.json`](../calibration/camera_params.json)
- **Focal Length $(f_x, f_y)$**: `(1500.50, 1500.59) px`
- **Principal Point $(c_x, c_y)$**: `(991.25, 596.00) px`
- **Distortion Coefficients $D$**: `[-0.09048, -0.05970, -0.00352, -0.00715, 0.37573]`
- **Visual Diagnostics**: [`calibration/diagnostics/undistortion_proof_grid.png`](../calibration/diagnostics/undistortion_proof_grid.png) and [`calibration/diagnostics/reprojection_error_plot.png`](../calibration/diagnostics/reprojection_error_plot.png)

### 2. Dataset Evidence
- **Total Acquired Images**: 80 high-resolution images ($1920 \times 1080\text{ px}$)
- **Train Split Count**: 56 samples (70.0% - [`dataset/splits/train.txt`](../dataset/splits/train.txt))
- **Validation Split Count**: 16 samples (20.0% - [`dataset/splits/val.txt`](../dataset/splits/val.txt))
- **Test Split Count**: 8 samples (10.0% - [`dataset/splits/test.txt`](../dataset/splits/test.txt), frozen prior to training)
- **Labeling Format**: Lossless 8-bit single-channel PNG binary segmentation masks (`{0, 255}`)
- **Mean Object Foreground Area**: `104,649 px`
- **Mean Bounding Box Aspect Ratio**: `1.585` (matches nominal physical ratio $85.5\text{ mm} / 54.0\text{ mm} = 1.583$)
- **Visual Diagnostics**: [`dataset/diagnostics/sample_inspection_overlay.png`](../dataset/diagnostics/sample_inspection_overlay.png) and [`dataset/diagnostics/dataset_distribution.png`](../dataset/diagnostics/dataset_distribution.png)

### 3. Model Training Evidence
- **Architecture**: DeepLabV3+ (Encoder: ResNet-34 ImageNet pretrained, Decoder: ASPP)
- **Framework Compliance**: Pure PyTorch (`segmentation_models_pytorch`). Non-YOLO, Non-Roboflow verified.
- **Epochs Run**: 15 epochs (Batch size 8, AdamW + CosineAnnealingLR)
- **Best Validation IoU**: `0.9364`
- **Saved Weights Checkpoint**: `models/checkpoints/best_model.pth`
- **SHA-256 Hash**: `80dbde96c5fce2e870f3b5a56c164d6ceb617ac4da7fb3adc22d3556f9df0395`
- **Isolated Test Split Metrics (`dataset/splits/test.txt`)**:
  - **Mask IoU**: `0.9423` (Target: $\ge 0.88$ — **MET**)
  - **mAP@0.5**: `1.0000` (Target: $\ge 0.92$ — **MET**)
  - **mAP@0.5:0.95**: `0.9375`
  - **Precision**: `0.9589`
  - **Recall**: `0.9820`
- **Visual Diagnostics**: [`models/diagnostics/test_predictions_overlay.png`](../models/diagnostics/test_predictions_overlay.png) and [`models/diagnostics/training_curves.png`](../models/diagnostics/training_curves.png)

### 4. Metrology Accuracy Summary
- **Instances Benchmarked**: 
- **Mean Absolute Error (MAE)**: 
- **Mean Percentage Error (MPE)**: 
- **Worst-Case Deviation**: 

### 5. Incomplete Items Declared
*(Honest declaration of any items or secondary extensions not reached within the 8-hour window, along with root-cause reasons.)*
- None declared yet.
