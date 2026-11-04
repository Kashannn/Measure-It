# Serena Memory: Loom Viva Defense (Questions K1 to K4)

These answers are structured for the candidate's spoken response in the 5-minute Loom recording and the follow-up technical interview.

---

### Question K1:
> **"Which file and line turns pixels into millimetres? What has to be true about the scene for that conversion to be valid?"**

- **Code Location**: 
  - [`inference/geometry.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/inference/geometry.py#L82-L85) lines 82–85:
    ```python
    width_mm = w_px / pixels_per_mm
    height_mm = h_px / pixels_per_mm
    ```
  - The spatial scale factor `pixels_per_mm` is computed in [`inference/detector.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/inference/detector.py#L87-L95) lines 87–95:
    ```python
    pixels_per_mm = mean_edge_px / self.marker_size_mm  # 50.0 mm
    ```
- **Scene Invariants That Must Be True**:
  1. **Strict Coplanarity**: The target object's top measurement face and the reference marker must lie in the exact same physical depth plane ($Z = \text{constant}$). If the reference is elevated or recessed, perspective foreshortening invalidates the uniform scaling ratio.
  2. **Orthogonal/Near-Orthogonal Viewing Angle**: The optical axis must be perpendicular to the plane (tilt $\le 15^\circ$). Beyond $20^\circ$, perspective foreshortening breaks the scalar ratio and causes error to explode ($> 48\text{ mm}$).
  3. **Zero Distortion**: Radial and tangential lens distortion must be rectified prior to measurement so that straight physical edges project as straight lines.

---

### Question K2:
> **"Where in your code is undistortion applied, and how much did skipping it change your measurement error? Use your own numbers."**

- **Code Location**: 
  - [`inference/pipeline.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/inference/pipeline.py#L100-L105) inside `undistort_image()`, calling `cv2.undistort()` with camera matrix $K$ and distortion coefficients $D$ loaded from [`calibration/camera_params.json`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/calibration/camera_params.json). Runs before semantic segmentation and fiducial detection.
- **Empirical Numbers ([`measurement/ablation_results.json`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/measurement/ablation_results.json))**:
  - Across our 12 held-out benchmark specimens:
    - **Raw Distorted Images (Without Undistortion)**: MAE = **$3.169\text{ mm}$**.
    - **Undistorted Images (Pinhole Model)**: MAE = **$1.433\text{ mm}$**.
    - **Error Reduction Delta**: **$1.735\text{ mm}$** (**$54.8\%$ empirical error reduction**).
    - For objects placed near the outer frame edges ($> 400\text{ px}$ from optical center, e.g. `img_0070.jpg`), raw barrel distortion added up to **$4.60\text{ mm}$** of error.

---

### Question K3:
> **"What is the model architecture, why was it chosen over alternatives, and what are its failure modes?"**

- **Architecture**: 
  - Custom **DeepLabV3+** with a ResNet-34 backbone and Atrous Spatial Pyramid Pooling (ASPP) decoder implemented in pure PyTorch (`segmentation_models_pytorch`).
  - Strict compliance: Non-YOLO, Non-Roboflow.
- **Why Chosen**: 
  - ASPP uses atrous (dilated) convolutions at multiple dilation rates (6, 12, 18) to capture multi-scale context without reducing spatial resolution. Unlike standard UNets or FCNs which round sharp rectangular corners, DeepLabV3+ preserves sub-millimetre edge boundaries.
  - Achieved **0.9423 Test Mask IoU** and **1.0000 mAP@0.5** on the isolated test set.
- **Failure Modes**:
  - Severe camera tilt ($\ge 20^\circ$) causing foreshortened aspect ratios and boundary blurring. Handled by defensive refusal guardrails in `inference/guardrails.py`.

---

### Question K4:
> **"If given another day, what would you improve and what would you definitely NOT do?"**

- **What I Would Improve**:
  1. **Sub-Pixel Active Contour Edge Refinement**: Run a Snakes (active contour) or GrabCut energy minimization on the neural mask boundary to refine edges down to sub-pixel accuracy, reducing MAE from $1.43\text{ mm}$ to $< 0.40\text{ mm}$.
  2. **Planar Homography Rectification**: Estimate a $3 \times 3$ homography matrix $H$ from the 4 ArUco corners to warp oblique captures into an orthographic top-down perspective, allowing reliable measurements up to $30^\circ$ tilt.
- **What I Would Definitely NOT Do**:
  - I would **NOT** replace semantic segmentation with bounding box object detection (e.g. Faster R-CNN or oriented bounding boxes). Bounding boxes cannot capture rounded corners or chamfered edges, and when parts are rotated, rectangular boxes capture background clutter, introducing massive systematic error.
