# ⚖️ Architecture Decision Records (ADRs)

This file documents critical architectural decisions, context, trade-offs, and rationale for **Measure-It**.

---

## ADR-001: Prohibition of Ultralytics YOLO and Roboflow Frameworks

- **Status:** Accepted ✅
- **Date:** 2026-11-04
- **Context:**
  Section 4.2 and Section 9 of the assessment guidelines strictly forbid Ultralytics YOLO models and Roboflow hosted/prepackaged models. Violating this triggers automatic rejection.
- **Decision:**
  Use native PyTorch with `segmentation_models_pytorch` to build a modular DeepLabV3+ network with a standard ResNet-34 ImageNet-pretrained encoder.
- **Consequences:**
  - Full compliance with assessment constraints.
  - Transparent training loop that can be debugged and explained from first principles in the technical interview.
  - Independence from third-party cloud APIs and vendor-locked frameworks.

---

## ADR-002: ArUco Marker Fiducial as Metric Reference Over Circular Coins

- **Status:** Accepted ✅
- **Date:** 2026-11-04
- **Context:**
  Metric metrology requires a co-planar reference of known physical dimensions. Candidates often use coins (e.g., a 10-cent coin) or credit cards. However, coins suffer from specular glare, edge wear, and circular ellipse fitting ambiguities under slight perspective.
- **Decision:**
  Standardize on a printed, rigid OpenCV ArUco square marker (Dictionary: `DICT_4X4_50`, Marker ID: 0, $50.0 \times 50.0\text{ mm}$ measured with digital caliper).
- **Consequences:**
  - Robust corner detection with sub-pixel optimization (`cv2.cornerSubPix`).
  - Distinct binary pattern provides deterministic detection invariant to lighting conditions.
  - Aspect ratio of the 4 detected corners allows instant detection of camera perspective tilt.

---

## ADR-003: Rotated Minimum Bounding Box (`cv2.minAreaRect`) Over Axis-Aligned Box

- **Status:** Accepted ✅
- **Date:** 2026-11-04
- **Context:**
  In real-world inspection, objects are rarely placed in perfect horizontal or vertical alignment with the camera sensor. Standard axis-aligned bounding boxes (`cv2.boundingRect`) overestimate dimensions by up to $41.4\%$ when an object is rotated $45^\circ$.
- **Decision:**
  Implement `cv2.minAreaRect` on the largest segmentation mask contour to compute the oriented bounding rectangle.
- **Consequences:**
  - True physical dimensions (width and height) are recovered regardless of object rotation angle.
  - Directly matches the physical caliper measurement orientation.

---

## ADR-004: Pre-Commit Frozen Test Split (`dataset/splits/test.txt`)

- **Status:** Accepted ✅
- **Date:** 2026-11-04
- **Context:**
  Rules 5.1 and 5.5 require that the 10% test split be chosen and committed *before* training starts, and never touched for hyperparameter tuning or checkpoint selection. Assessors examine Git commit history to verify this order.
- **Decision:**
  Create and commit `dataset/splits/test.txt` along with the initial architecture documents before writing any training or modeling code.
- **Consequences:**
  - Verifiable zero test set leakage in Git history.
  - Guaranteed credibility of the reported test metrics (mIoU and mAP).

---

## ADR-005: Combined Binary Cross-Entropy + Dice Loss

- **Status:** Accepted ✅
- **Date:** 2026-11-04
- **Context:**
  Standard cross-entropy loss suffers from pixel imbalance when the target object occupies a small fraction of the image frame, often resulting in smooth, eroded contours.
- **Decision:**
  Use a compound loss: $\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$.
- **Consequences:**
  - BCE penalizes pixel-level misclassifications uniformly.
  - Dice loss directly optimizes the intersection-over-union metric, preserving sharp orthogonal corners essential for accurate physical dimension measurement.

---

## ADR-006: Defensive Refusal (Knowing When Not to Answer)

- **Status:** Accepted ✅
- **Date:** 2026-11-04
- **Context:**
  Section 3 Item 9 and Section 2 dictate: *"A measurement system that reports a wrong number confidently is worse than one that reports nothing."*
- **Decision:**
  Implement an explicit guardrail module that refuses to output numbers if:
  1. ArUco corner perspective tilt exceeds $20^\circ$.
  2. Mask sigmoid confidence is below $0.85$.
  3. Reference fiducial is partially occluded.
- **Consequences:**
  - Zero catastrophic hallucinated dimensions.
  - Transparent error codes and explanations returned to the user instead of misleading numbers.
