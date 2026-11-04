# 🎯 GSD (Get Stuff Done) Documentation System

Welcome to the **.gsd** documentation and state tracking system for **Measure-It** (`measure_it_cv`).

This directory serves as the single source of truth for AI agents, developers, and reviewers. It implements a disciplined, spec-driven development methodology: **Discuss → Plan → Execute → Verify**.

---

## 📂 Directory Index

| File | Purpose |
| :--- | :--- |
| [**STATE.md**](./STATE.md) | **Current Project State:** Active milestones, module implementation registry, health metrics, and active blockers. |
| [**FLOW.md**](./FLOW.md) | **Optical Metrology & CV Flows:** End-to-end inference flow, camera calibration pipeline, training loop, and defensive rejection guardrails. |
| [**CHANGELOG.md**](./CHANGELOG.md) | **Chronological History:** Comprehensive log of versions, architectural milestones, commits, and experiments. |
| [**ROADMAP.md**](./ROADMAP.md) | **Execution Timeline:** Immediate priorities, 8-hour sprint budget allocation, and delivery checkpoints. |
| [**ARCHITECTURE.md**](./ARCHITECTURE.md) | **System Architecture:** Modular pipeline layout, pinhole camera model, DeepLabV3+ neural network design, and geometric algorithms. |
| [**DECISIONS.md**](./DECISIONS.md) | **Architecture Decision Records (ADRs):** Rationale behind critical computer vision, ML model, and metrology choices. |

---

## 🛠️ Operating Principles

1. **Keep in Sync**: Whenever an architectural change, new feature, or calibration benchmark occurs, update the relevant files in `.gsd/`.
2. **Context Integrity**: Before starting model training or metrology script execution, consult [STATE.md](./STATE.md) and [ARCHITECTURE.md](./ARCHITECTURE.md) to preserve mathematical and engineering invariants.
3. **Spec-Driven**: Implementation must strictly follow [DECISIONS.md](./DECISIONS.md) (e.g., zero YOLO/Roboflow dependencies) and conform to error budgets in [ROADMAP.md](./ROADMAP.md).
