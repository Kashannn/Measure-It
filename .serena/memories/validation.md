# Serena Memory: Validation, Caliper Benchmarking & Ablation Protocols

## Caliper Verification Protocol (10+ Instances)
- Ground truth instrument: Digital Caliper ($\pm 0.02\text{ mm}$ vernier accuracy).
- Each test object placement is manually measured 3 times with caliper; the mean is stored as Ground Truth (GT).
- System execution is logged with raw standard output containing:
  - Instance ID
  - Ground Truth Width ($W_{\text{GT}}$) and Height ($H_{\text{GT}}$)
  - System Predicted Width ($W_{\text{pred}}$) and Height ($H_{\text{pred}}$)
  - Absolute Error: $e_{\text{abs}} = |y_{\text{GT}} - y_{\text{pred}}|$
  - Percentage Error: $e_{\text{pct}} = \frac{|y_{\text{GT}} - y_{\text{pred}}|}{y_{\text{GT}}} \times 100\%$

## Ablation Study: Undistortion Impact (Item 6)
To prove empirically why camera calibration matters:
1. Run pipeline on `img_raw` (bypassing `cv2.undistort`).
2. Run pipeline on `img_undistorted`.
3. Tabulate side-by-side error columns:
   - Error on Raw (mm)
   - Error on Undistorted (mm)
   - Improvement Delta: $\Delta = e_{\text{raw}} - e_{\text{undistorted}}$
4. Objects located near the periphery of the image sensor experience significant radial compression (barrel distortion), causing raw scale estimation to underestimate dimensions by $2\text{ mm}$ to $6\text{ mm}$.
