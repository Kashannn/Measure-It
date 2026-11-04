# Serena Memory: Computer Vision & Metrology Pipeline

## Mathematical Foundations

### 1. Camera Pinhole Model & Lens Distortion
The pinhole camera projection transforms 3D world coordinates $[X_w, Y_w, Z_w]^T$ into 2D pixel coordinates $[u, v]^T$:
$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} R & t \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
Where $K$ is the intrinsic camera matrix:
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Lens distortion is modeled using radial ($k_1, k_2, k_3$) and tangential ($p_1, p_2$) coefficients:
$$x_{\text{distorted}} = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2 x^2)$$
$$y_{\text{distorted}} = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y^2) + 2 p_2 x y$$
Undistortion is solved iteratively using `cv2.undistort` with optimal new camera matrix to preserve rectilinear geometry.

### 2. Metric Conversion (Pixels to Millimetres)
Under the coplanar assumption ($Z = 0$), the ratio of physical distance to image distance is constant across the plane:
$$\text{Scale} = \frac{\text{Perimeter}_{\text{ArUco, px}}}{4 \times \text{Dimension}_{\text{ArUco, mm}}} \quad [\text{pixels per mm}]$$

For an object contour:
1. Compute oriented bounding box:
   $$\text{box} = \text{cv2.minAreaRect}(\text{contour})$$
2. Extract box dimensions in pixels: $(w_{\text{px}}, h_{\text{px}})$.
3. Convert to millimeters:
   $$W_{\text{mm}} = \frac{w_{\text{px}}}{\text{Scale}}, \quad H_{\text{mm}} = \frac{h_{\text{px}}}{\text{Scale}}$$

### 3. Defensive Guardrails
- **Tilt Angle Check**: Check if ArUco sides deviate in length by $> 8\%$, which indicates excessive camera pitch/roll.
- **Confidence Check**: Reject inference if mean mask sigmoid activation $< 0.85$.
- **Boundary Check**: Reject object if mask touches within 10 pixels of the sensor boundary.
