# Loom Video Recording Script & Viva Defense

> **System**: Measure-It Optical Metrology System  
> **Candidate**: Kashan Ashraf  
> **Assessment**: AI / Computer Vision Technical Assessment — xis.ai  
> **Target Duration**: Strictly $\le 5:00$ minutes (4:45 target)  
> **Format**: Screen share with candidate webcam picture-in-picture  

---

## ⏱️ Video Timeline Breakdown

```
[0:00 - 0:40] Introduction, Problem Statement & Metrology Setup
[0:40 - 1:30] Camera Calibration & Lens Undistortion Proof (Question K2)
[0:30 - 2:30] Custom DeepLabV3+ Segmentation Model & Test Split (Question K3)
[2:30 - 3:30] Live CLI Execution & Dimension Derivation (Question K1)
[3:30 - 4:15] Tilt Stress Test & Defensive Refusal Guardrails
[4:15 - 4:55] Next Steps: What to Do & What NOT to Do (Question K4)
[4:55 - 5:00] Closing & Pull Request Details
```

---

## 🎬 Word-for-Word Speaking Script

### 1. Introduction & Setup (0:00 – 0:40)
**[Webcam full or PIP, Screen showing Measure-It repository]**

> *"Hello Mahnoor and the xis.ai engineering team. My name is Kashan Ashraf. Today I am presenting **Measure-It**, an optical metrology system that photographs an industrial specimen and computes its real physical dimensions in millimetres using a calibrated camera, a custom non-YOLO semantic segmentation model, and an ArUco metric reference.*
>
> *To guarantee complete scientific integrity, this project followed a strict protocol: our project plan, measurable objectives, and isolated test split were committed to Git **before** running any model training or calibration. Let's look at how the system works."*

---

### 2. Camera Calibration & Undistortion (0:40 – 1:30)
**[Screen showing `calibration/diagnostics/undistortion_proof_grid.png` and `calibration/camera_params.json`]**
*(Addresses **Question K2**)*

> *"Real-world camera lenses suffer from optical barrel distortion where straight lines bend outward, especially toward sensor edges. 
> 
> In **Question K2**, how does camera undistortion work, where does it live in the pipeline, and why does it matter?
>
> In our code, undistortion lives right at the entry point of the pipeline in [`inference/pipeline.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/inference/pipeline.py#L100-L105) inside `undistort_image()`. We calibrated our camera using 23 checkerboard poses with OpenCV's pinhole model, achieving a sub-pixel mean reprojection error of **0.508 pixels**.
>
> Undistortion matters immensely for metrology: as proven in our ablation study in [`measurement/ablation_undistort.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/measurement/ablation_undistort.py), measuring specimens without undistortion produced a Mean Absolute Error of **3.17 mm**. With calibrated undistortion enabled, that error dropped to **1.43 mm**—an empirical **54.8% error reduction**."*

---

### 3. Custom Segmentation Model & Test Split (1:30 – 2:30)
**[Screen showing `models/diagnostics/test_predictions_overlay.png` and `models/evaluate.py`]**
*(Addresses **Question K3**)*

> *"Per the assessment guidelines, Ultralytics YOLO and Roboflow models were strictly prohibited. 
>
> In **Question K3**, what is the model architecture, why was it chosen, and what are its failure modes?
>
> We implemented a custom **DeepLabV3+** architecture built with a ResNet-34 encoder and an Atrous Spatial Pyramid Pooling (ASPP) decoder in pure PyTorch ([`models/model.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/models/model.py)). We chose DeepLabV3+ because the dilated convolutions in the ASPP module preserve high-resolution spatial boundaries while capturing global context—crucial for sub-millimetre edge delineation, whereas simple FCNs or standard UNets blur corners.
>
> We evaluated the model on the strictly isolated test split (`dataset/splits/test.txt`), achieving a **Test Mask IoU of 0.9423** (surpassing the $\ge 0.88$ requirement) and **mAP@0.5 of 1.0000**.
>
> Its primary failure mode occurs under severe out-of-plane perspective tilt ($> 20^\circ$) where steep shadows and foreshortening degrade boundary contrast. This brings us to how we extract dimensions."*

---

### 4. Single-Command Metrology & Dimension Derivation (2:30 – 3:30)
**[Screen showing terminal running `python measure.py --image dataset/images/img_0073.jpg`]**
*(Addresses **Question K1**)*

> *"Let's run our single-command CLI live in the terminal:
> `python measure.py --image dataset/images/img_0073.jpg`
>
> The system loads the image, undistorts the lens, detects the $50\text{ mm}$ ArUco marker, segments the specimen, and produces our diagnostic overlay. Here, the ground-truth specimen is $54.0 \times 85.5\text{ mm}$. Our system computes **54.90 mm width** and **88.46 mm height** with 0.962 confidence.
>
> This directly answers **Question K1: Which file and line turns pixels into millimetres?**
>
> In [`inference/geometry.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/inference/geometry.py#L82-L85) lines 82 to 85, inside `MetricContourExtractor.extract_dimensions()`:
> ```python
> width_mm = w_px / pixels_per_mm
> height_mm = h_px / pixels_per_mm
> ```
> The spatial scale ratio `pixels_per_mm` is computed in [`inference/detector.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/inference/detector.py#L87-L95) by dividing the detected sub-pixel ArUco perimeter edge in pixels by the known physical dimension of $50.0\text{ mm}$.
>
> In our digital caliper benchmark across 12 held-out specimens ([`measurement/benchmark_accuracy.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/measurement/benchmark_accuracy.py)), we achieved **MAE = 1.43 mm** ($\le 1.50\text{ mm}$) and **MPE = 1.97%** ($\le 2.0\%$), satisfying Measurable Objective 1."*

---

### 5. Failure Modes & Defensive Guardrails (3:30 – 4:15)
**[Screen showing `measurement/diagnostics/tilt_degradation_curve.png` and `inference/guardrails.py`]**

> *"In real industrial metrology, an algorithm must know when to refuse to answer. 
>
> We performed a tilt stress test from $0^\circ$ to $45^\circ$ ([`measurement/stress_test_tilt.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/measurement/stress_test_tilt.py)). At tilt angles $\ge 20^\circ$, perspective foreshortening breaks the coplanar assumption and error explodes beyond $48\text{ mm}$.
>
> Therefore, we implemented defensive guardrails in [`inference/guardrails.py`](file:///Users/kashanashraf/Documents/GitHub/Measure-It/inference/guardrails.py). If the image exhibits severe tilt (marker aspect skew $> 1.10$), excessive blur (Laplacian variance $< 80$), poor lighting ($< 40$ or $> 240$), or low model confidence ($< 0.85$), the system refuses the measurement and exits with **Code 4 (`UNTRUSTWORTHY_INPUT_REJECTED`)** rather than returning a corrupt measurement."*

---

### 6. Extensions: What to Do & What NOT to Do (4:15 – 4:55)
**[Screen showing `docs/Plan.md` or candidate camera]**
*(Addresses **Question K4**)*

> *"Finally, answering **Question K4: If given another day, what would you improve and what would you definitely NOT do?**
>
> **What I would improve:** I would implement sub-pixel active contour energy minimization (Snakes or GrabCut boundary refinement) initialized from the neural mask. That would refine edge placement to sub-pixel precision, dropping our MAE from $1.4\text{ mm}$ down to $< 0.4\text{ mm}$. Second, I would add a homography rectification warp based on the 4 ArUco corners to restore planarity for tilts up to $25^\circ$.
>
> **What I would definitely NOT do:** I would NOT replace semantic segmentation with bounding box object detection (such as Faster R-CNN or oriented bounding box detectors). Bounding boxes cannot capture rounded corners or chamfered edges, and when parts are placed at arbitrary angles, rectangular bounding boxes encapsulate irrelevant background shadows, introducing substantial systematic error."*

---

### 7. Closing & Submission (4:55 – 5:00)
**[Screen showing GitHub Pull Request on branch `dev-test01`]**

> *"All code, calibration files, test splits, model checkpoints, and reproducible scripts are committed on branch `dev-test01` with an open PR into `main`. Thank you for your time, and I look forward to the technical discussion!"*
