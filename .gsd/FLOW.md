# 🔄 Optical Metrology & System Flows

This document details the operational execution flows, state machines, and mathematical transformations inside **Measure-It**.

---

## 1. Single Command Metrology Execution Flow (`inference/measure.py`)

```mermaid
sequenceDiagram
    autonumber
    actor User as Engineer / Camera
    participant CLI as inference/measure.py
    participant Undistort as Lens Rectifier (cv2.undistort)
    participant Model as DeepLabV3+ Neural Net
    participant ArUco as ArUco Scale Detector
    participant Metrology as minAreaRect Engine
    participant Guard as Guardrail Filter

    User->>CLI: python inference/measure.py --image input.jpg
    CLI->>Undistort: Load camera_params.json & Undistort Image
    Undistort-->>CLI: Undistorted Rectilinear Image

    par Inference & Reference Detection
        CLI->>Model: Forward pass on Undistorted Image
        Model-->>CLI: Binary Mask & Mean Confidence Score
    and
        CLI->>ArUco: Detect DICT_4X4_50 Fiducial Corners
        ArUco-->>CLI: 4 Sub-pixel Corners & Edge Lengths
    end

    CLI->>Guard: Verify Coplanarity, Marker Aspect & Mask Confidence
    alt Guardrail Violated
        Guard-->>CLI: REJECT (Tilted plane / Low confidence)
        CLI-->>User: Refusal output: Error code & explanation (No guesswork)
    else Guardrail Passed
        Guard-->>CLI: APPROVED
        CLI->>Metrology: Extract Mask Contour & Run cv2.minAreaRect()
        Metrology->>Metrology: Compute PPM = Mean_Marker_Edge_px / 50.0 mm
        Metrology->>Metrology: Width_mm = Box_W_px / PPM, Height_mm = Box_H_px / PPM
        Metrology-->>CLI: Width (mm), Height (mm), Confidence
        CLI->>User: JSON output + Annotated diagnostic overlay image
    end
```

---

## 2. Camera Calibration Flow (`calibration/calibrate.py`)

```mermaid
flowchart TD
    A["Acquire 25+ Checkerboard Photos (Locked Focus & Zoom)"] --> B["Iterate through each calibration image"]
    B --> C["Detect inner corners via cv2.findChessboardCorners()"]
    C -->|Success| D["Refine to sub-pixel coordinates via cv2.cornerSubPix()"]
    C -->|Failed| E["Discard blurry/partial frame"]
    D --> F["Collect 3D Object Points & 2D Image Points across all views"]
    F --> G["Compute Camera Matrix K and Distortion Coefficients D"]
    G --> H["Calculate Mean Reprojection Error (MRE)"]
    H --> I{"Is MRE < 0.40 pixels?"}
    I -->|Yes| J["Save to calibration/camera_params.json"]
    I -->|No| K["Investigate warped frames or inconsistent focal lengths"]
```

---

## 3. Training & Evaluation Flow (`models/`)

```mermaid
flowchart TD
    A["Physical Labeled Images (Undistorted first)"] --> B["Split into Train (70%), Val (20%), Test (10%)"]
    B --> C["Verify Test Split matches dataset/splits/test.txt exactly"]
    C --> D["PyTorch Custom Dataset with Albumentations"]
    D --> E["Forward pass DeepLabV3+ with ResNet-34 Backbone"]
    E --> F["Compute Combined BCE + Dice Loss"]
    F --> G["Backpropagate & AdamW Step"]
    G --> H["Evaluate Validation IoU at epoch end"]
    H --> I{"Is Val IoU highest?"}
    I -->|Yes| J["Save Checkpoint: models/checkpoints/best_model.pth"]
    I -->|No| K["Continue Training / Early Stopping counter"]
    J --> L["Final Evaluation on Isolated Test Split (test.txt)"]
    L --> M["Output: Mask IoU, mAP@0.5, Precision, Recall"]
```

---

## 4. Defensive Guardrail Decision Table

| Failure Trigger | Detection Mechanism | System Action | Exit Code |
|---|---|---|---|
| **No Target Object** | Mask area $< 500\text{ px}$ | Refuse to output dimensions; advise retaking photo | `ERR_OBJECT_NOT_FOUND` |
| **No Reference Marker** | `cv2.aruco.detectMarkers` returns zero markers | Refuse to output dimensions; scale cannot be determined | `ERR_REFERENCE_NOT_FOUND` |
| **Severe Perspective Tilt** | Ratio of max/min ArUco sides $> 1.08$ ($> 20^\circ$ tilt) | Refuse to answer; perspective foreshortening invalidates scale | `ERR_TILT_EXCEEDED` |
| **Low Model Confidence** | Mean sigmoid prediction activation on mask $< 0.85$ | Refuse to answer; segmentation boundary uncertain | `ERR_LOW_CONFIDENCE` |
| **Object Near Frame Edge** | Bounding box within 10 pixels of image border | Refuse to answer; partial truncation suspected | `ERR_EDGE_TRUNCATION` |
