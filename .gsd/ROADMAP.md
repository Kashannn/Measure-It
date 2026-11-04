# 🗺️ Project Roadmap & 8-Hour Execution Budget

**Project:** Measure-It — Optical Metrology System  
**Assessment:** xis.ai Technical Assessment v2.0  
**Budget Window:** 8 Working Hours (24-Hour Submission Window)  

---

## 1. 8-Hour Budget Timeline

| Phase | Milestone | Duration | Planned Output | Status |
| :---: | :--- | :---: | :--- | :---: |
| **Phase 0** | **Groundwork & Spec Architecture** | 0.5 hr | `docs/*`, `.gsd/*`, `.serena/*`, frozen `splits/` committed first | ✅ **Complete** |
| **Phase 1** | **Camera Calibration & Undistortion** | 1.0 hr | 25+ checkerboard captures, $K$ & $D$ matrices, MRE $< 0.40\text{ px}$ | ⏳ Pending Capture |
| **Phase 2** | **Dataset Acquisition & CVAT Masking** | 1.5 hrs | 80 photos undistorted, binary masks annotated, splits frozen | ⏳ Pending Data |
| **Phase 3** | **DeepLabV3+ Training & Test Eval** | 1.0 hr | Checkpoint saved, test split IoU $\ge 0.88$, mAP $\ge 0.92$ | ⏳ Pending Train |
| **Phase 4** | **Inference CLI (Image $\to$ mm)** | 1.5 hrs | Single command `measure.py` with ArUco scaling & overlay | ⏳ Pending CLI |
| **Phase 5** | **Caliper Benchmark & Undistort Proof**| 1.0 hr | 10+ items measured vs digital caliper, raw vs undistorted table | ⏳ Pending Test |
| **Phase 6** | **Tilt Breaking Test & Rejection Logic**| 0.5 hr | Tilt error curve ($0^\circ \to 45^\circ$), rejection guardrail tests | ⏳ Pending Test |
| **Phase 7** | **Loom Video (<5 min) & PR Submission**| 1.0 hr | Screen+caliper demo, K1–K4 answers, open PR into `main` | ⏳ Pending Video |

---

## 2. Assessment Deliverable Matrix (Section 3 of Specification)

- [ ] **Item 1: Calibrated camera** (20+ checkerboard images, camera matrix & distortion coefficients saved, MRE reported).
- [ ] **Item 2: Own labeled dataset** (70+ real images with calibrated camera, undistorted, 70/20/10 split committed).
- [ ] **Item 3: Trained segmentation model** (Not YOLO, not Roboflow. Test split evaluated: IoU, mAP@0.5, mAP@0.5:0.95, Precision, Recall).
- [ ] **Item 4: One command, image to millimetres** (`measure.py` undistorts, segments, detects reference, outputs mm width/height and overlay).
- [ ] **Item 5: Accuracy against a caliper** (10+ instances measured by hand vs system, MAE and MPE reported).
- [ ] **Item 6: Proof that undistortion matters** (Measure same instances on raw vs undistorted, side-by-side error delta).
- [ ] **Item 7: Speed or accuracy target** (MO-1 & MO-2 defined and measured).
- [ ] **Item 8: One condition that breaks it** (Camera perspective tilt degradation curve).
- [ ] **Item 9: Knowing when not to answer** (Refusal guardrails for tilted plane, low confidence, occluded reference).
- [ ] **Item 10: Decision record inside Plan** (Chosen vs rejected approaches and trade-offs).
