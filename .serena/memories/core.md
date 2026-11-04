# Serena Memory: Core Architecture & Assessment Rules

## Project Identity
- **Assessment**: AI / Computer Vision Technical Assessment — `xis.ai`
- **Candidate**: Kashan Ashraf
- **Objective**: Develop an optical metrology system measuring real-world dimensions (mm) from a single photo with calibrated camera and custom segmentation model.

## Non-Negotiable Invariants & Rules
1. **Model Constraints**:
   - MUST NOT use Ultralytics YOLO or Roboflow models under any circumstance.
   - MUST use standard PyTorch / `segmentation_models_pytorch` (DeepLabV3+ with ResNet-34 backbone).
2. **Git Discipline**:
   - Working branch is strictly `dev-test01`.
   - Never commit everything in a single dump.
   - Never commit code before the Plan and test split are committed.
   - Clear imperative commit messages (no generic "wip" or "fix").
3. **Data Integrity**:
   - Every single image must be captured in the physical world by the candidate.
   - Test split (10%) must remain completely untouched during model training and checkpoint selection.
4. **Metrology Validity**:
   - Target object and metric reference (ArUco marker) must reside in the exact same physical plane.
   - Images must undergo camera matrix undistortion prior to segmentation and dimension calculation.
   - Camera focus, zoom, and sensor resolution must remain rigidly locked across all captures.
