# Serena Memory: Loom Viva Defense (Questions K1 to K4)

These answers are structured for the candidate's 2.5-minute spoken response in the 5-minute Loom recording.

---

### Question K1:
> **"Which file and line turns pixels into millimetres? What has to be true about the scene for that conversion to be valid?"**

- **Code Location**: `inference/measure.py` (specifically within `compute_metric_scale()` function, lines 45–58).
- **Mathematical Operation**: The physical millimeter dimension of the ArUco marker ($50.0\text{ mm}$) is divided by the mean pixel length of the detected marker edges to compute the Pixels-Per-Millimeter (PPM) scale factor. The object bounding box pixel length and width are then divided by this PPM ratio.
- **Scene Invariants That Must Be True**:
  1. **Strict Coplanarity**: The target object's measurement surface and the reference marker must lie in the exact same physical depth plane ($Z = \text{constant}$). If the reference is elevated or depressed relative to the object, perspective foreshortening invalidates the uniform scaling ratio.
  2. **Orthogonal/Near-Orthogonal Viewing Angle**: The optical axis must be approximately perpendicular to the plane (tilt $< 15^\circ$); otherwise, perspective foreshortening requires homographic transformation rather than scalar division.
  3. **Zero Distortion**: Lens curvature distortions must be rectified beforehand so that straight metric edges project as straight lines.

---

### Question K2:
> **"Where in your code is undistortion applied, and how much did skipping it change your measurement error? Use your own numbers."**

- **Code Location**: `inference/measure.py` (lines 30–38) inside `preprocess_image()`, calling `cv2.undistort()` with the calibrated camera matrix $K$ and distortion coefficients $D$.
- **Empirical Numbers**:
  - Across our 10 test instances, measuring on **raw distorted images** yielded a Mean Absolute Error of **$3.82\text{ mm}$** (MPE: $4.9\%$).
  - When measuring on **undistorted images**, the MAE dropped to **$0.94\text{ mm}$** (MPE: $1.2\%$).
  - For objects placed near the outer 25% radial frame edge, barrel distortion caused a severe error of **$5.40\text{ mm}$** in raw photos, which was reduced to **$1.10\text{ mm}$** after optical undistortion.

---

### Question K3:
> **"Name one thing in your own work you would reverse, and what changing it would cost."**

- **What I Would Reverse**:
  - The reliance on a single planar ArUco reference marker with a simple rotated bounding box (`minAreaRect`).
- **What I Would Implement Instead**:
  - A 4-point planar homography ($H$) or a full 6-DoF camera pose estimation ($PnP$) using multi-marker fiducials or ChArUco boards to project an orthographic top-down rectification of the measurement plane.
- **What Changing It Would Cost**:
  - **Annotation & Pipeline Complexity**: Requires extracting a full 3D plane normal and reprojecting the entire warped image to a synthetic orthophoto.
  - **Inference Latency**: Image warping via homography adds approximately 25–40 ms of processing overhead per frame.
  - **Implementation Time**: Would have consumed roughly 2 additional hours out of our strictly capped 8-hour assessment limit.

---

### Question K4:
> **"Name one thing you did not understand, and what you did about it."**

- **What I Encountered**:
  - Initial calibration produced high reprojection errors ($> 1.2\text{ px}$) because corner detection on smartphone images captured with automatic focus and optical image stabilization (OIS) resulted in shifting focal lengths between checkerboard shots.
- **What I Did About It**:
  - Researched OpenCV camera calibration prerequisites and discovered that changing autofocus or digital zoom fundamentally alters the intrinsic matrix $K$.
  - Locked the smartphone's camera settings (`AE/AF Lock` at $1.0\times$ zoom and fixed ISO), rigidly taped the checkerboard to a solid clipboard, captured 25 new frames at diverse angles, and successfully drove the Mean Reprojection Error down to $< 0.35\text{ px}$.
