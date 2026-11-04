# 📜 Changelog

All notable changes, commits, experiments, and architectural transitions for **Measure-It** are documented in this file.

---

## [0.1.0] - 2026-11-04 (Commit: Initial Architecture Baseline)

### Added
- **Branch Initialization**: Created and switched to official assessment branch `dev-test01`.
- **Mandatory Assessment Documents**:
  - `docs/Plan.md`: 1-page timeline, 8-hour budget allocation, and Architectural Decision Records.
  - `docs/Objectives.md`: Formal measurable targets (MO-1: Caliper MAE $\le 1.5\text{ mm}$, MO-2: Mask IoU $\ge 0.88$, MO-3: Undistortion impact), plus hardware specifications.
  - `docs/Definition_of_Done.md`: Acceptance checklist with verification evidence links and logging tables.
- **Frozen Dataset Partitions**:
  - Committed `dataset/splits/test.txt` (10%), `dataset/splits/train.txt` (70%), and `dataset/splits/val.txt` (20%) prior to any model training code to guarantee zero test leakage.
- **Serena Agent System**:
  - Configured `.serena/project.yaml` and `.serena/memories/` (`core.md`, `cv_pipeline.md`, `validation.md`, `interview_defense.md`) for persistent MCP-driven agent memory and Loom viva answers (K1 to K4).
- **Graphify Integration**:
  - Added `graphify/README.md` and initialized semantic knowledge graph infrastructure via Graphify.
- **GSD System (`.gsd/`)**:
  - Created `.gsd/` folder with `README.md`, `STATE.md`, `ARCHITECTURE.md`, `FLOW.md`, `DECISIONS.md`, `ROADMAP.md`, and `CHANGELOG.md` following the spec-driven development framework.
- **Repository Metadata**:
  - Added comprehensive `ROADMAP.md`, `README.md`, `.gitignore`, and `requirements.txt`.
