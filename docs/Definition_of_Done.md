# Definition of Done (DoD) & Acceptance Checklist

> **Assessment**: AI / Computer Vision Technical Assessment — xis.ai  
> **Candidate**: Kashan Ashraf  
> **Branch**: `dev-test01`  
> **Rule**: Written *before* work starts. Ticked at the conclusion of work with raw evidence, numbers, or commit links beside each entry. An unticked box or a box ticked without evidence is counted as unverified.

---

## Acceptance Criteria Checklist

| Status | Verification Item | Target Standard | Evidence / Output Link |
|:---:|---|---|---|
| [ ] | **Camera Calibration Matrix & Parameters** | 20+ checkerboard images, intrinsic matrix $K$ and distortion coefficients $D$ saved to JSON, Mean Reprojection Error reported. | *Pending execution (Target: MRE < 0.40 px across 25+ images)* |
| [ ] | **Fixed Test Split Committed First** | Split lists created and committed to Git before training script is run. No test leakage. | *Committed in initial commit (Link to commit hash)* |
| [ ] | **Real Labeled Dataset** | 70+ real photographs captured with calibrated camera, undistorted, labeled with segmentation masks (70/20/10 split). | *Pending data collection (70+ real images, zero synthetic/downloaded)* |
| [ ] | **Trained Custom Segmentation Model** | Not YOLO, Not Roboflow. Trained on custom dataset. Evaluated on test split for IoU, mAP@0.5, mAP@0.5:0.95, Precision, Recall. | *Pending model training (DeepLabV3+ ResNet-34)* |
| [ ] | **Single-Command Pipeline (`inference/measure.py`)** | Takes one raw image path, undistorts, segments object, detects reference marker, prints width & height in mm with confidence score. | *Pending CLI implementation* |
| [ ] | **Clean Edge Case Handling** | Handles "Object not found" and "Reference not found" cleanly without crashing or false measurement. | *Pending test case suite* |
| [ ] | **Accuracy Benchmarking against Caliper** | 10+ real instances measured with system vs digital caliper. Per-instance table, MAE (mm) and MPE (%) reported. | *Pending caliper verification table* |
| [ ] | **Empirical Proof That Undistortion Matters** | Comparative measurement of same instances on raw vs undistorted images. Side-by-side error delta reported. | *Pending ablation run* |
| [ ] | **Measurable Objective 1 (MO-1 Met/Missed)** | MAE $\le 1.50\text{ mm}$ and MPE $\le 2.0\%$. Real machine output pasted. | *Pending evaluation* |
| [ ] | **Measurable Objective 2 (MO-2 Met/Missed)** | Test Mask IoU $\ge 0.88$, mAP@0.5 $\ge 0.92$. | *Pending evaluation* |
| [ ] | **Failure Mode Stress Test** | Breaking condition analyzed (camera tilt angle $0^\circ \to 45^\circ$) with quantitative degradation curve. | *Pending tilt stress test* |
| [ ] | **Defensive Rejection Threshold** | Rejection logic for untrustworthy inputs (low confidence, severe tilt, occluded reference) demonstrated. | *Pending rejection test* |
| [ ] | **Architectural Decision Record in Plan** | Chosen approach, rejected approaches, and the cost of rejection documented. | Documented in `docs/Plan.md` Section 4 |
| [ ] | **5-Minute Loom Video Recording** | $\le 5:00$ minutes. Live demo with physical caliper, code walkthrough, answering K1–K4 on camera. | *Pending recording upload URL* |
| [ ] | **Pull Request Opened** | PR created from branch `dev-test01` into `main` with summary and reproduction steps. | *Pending PR creation* |

---

## Execution Log & Evidence Notes

*(This section will be populated as each milestone is executed, providing raw terminal outputs, commit hashes, and metrics.)*

### 1. Calibration Evidence
- **Total Calibration Images**: 
- **Checkerboard Inner Grid**: $9 \times 7$ inner vertices (square size: $25.0\text{ mm}$)
- **Mean Reprojection Error**: 
- **Saved Parameters**: `calibration/camera_params.json`

### 2. Dataset Evidence
- **Total Acquired Images**: 
- **Train Split Count**: 
- **Validation Split Count**: 
- **Test Split Count**: 
- **Labeling Format**: 8-bit PNG binary segmentation masks

### 3. Model Training Evidence
- **Architecture**: DeepLabV3+ (Backbone: ResNet-34)
- **Epochs Run**: 
- **Best Validation IoU**: 
- **Test Set Metrics**:
  - Mask IoU:
  - mAP@0.5:
  - mAP@0.5:0.95:
  - Precision / Recall:

### 4. Metrology Accuracy Summary
- **Instances Benchmarked**: 
- **Mean Absolute Error (MAE)**: 
- **Mean Percentage Error (MPE)**: 
- **Worst-Case Deviation**: 

### 5. Incomplete Items Declared
*(Honest declaration of any items or secondary extensions not reached within the 8-hour window, along with root-cause reasons.)*
- None declared yet.
