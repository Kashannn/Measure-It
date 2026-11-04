# Serena Integration: Measure-It Metrology System

This directory houses the **Serena MCP (Model Context Protocol)** configuration and semantic memory store for the `Measure-It` codebase.

Serena empowers AI coding agents with symbol-level code understanding, project memory persistence, and semantic navigation across sessions.

---

## Directory Structure

```text
.serena/
├── README.md                 # Serena configuration overview
├── project.yaml              # Serena project parameters & language definitions
└── memories/                 # Long-lived persistent semantic memories
    ├── core.md               # Core metrology principles, constraints, and Git rules
    ├── cv_pipeline.md        # Mathematical formulation for calibration and scaling
    ├── validation.md         # Caliper benchmarking protocol and error metrics
    └── interview_defense.md  # Viva defense preparation for questions K1 to K4
```

---

## How Serena Uses These Memories

When an AI agent connects via Serena MCP:
1. Serena discovers the `.serena/memories/` graph, starting at `mem:core`.
2. Memories ensure that the agent never violates hard project invariants (e.g., strictly refusing YOLO/Roboflow architectures, enforcing locked test splits).
3. Agents maintain consistent reasoning across multi-step calibration and training iterations.
