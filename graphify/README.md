# Graphify Integration: Codebase Knowledge Graph

This directory contains configuration, scripts, and documentation for **Graphify** (`@Graphify-Labs/graphify`).

Graphify converts the entire `Measure-It` codebase, documentation, and architecture specifications into a persistent, queryable knowledge graph with community detection, interactive HTML visualization, and GraphRAG JSON output.

---

## What Graphify Provides

1. **Persistent Architectural Graph**: Connects mathematical calibration rules, dataset splits, DeepLabV3+ segmentation models, and metric conversion pipelines.
2. **Interactive Visualization (`graphify-out/graph.html`)**: Explore dependencies, modules, and cross-file references visually in any web browser.
3. **GraphRAG Traversal**: Query semantic concepts using breadth-first search (BFS) or depth-first search (DFS).
4. **Natural Language Explanations**: Ask plain-language questions about any module (e.g., `graphify explain "PinholeCamera"`).

---

## Quickstart Commands

### 1. Build Knowledge Graph
To scan the entire `Measure-It` repository and generate the interactive visualization and GraphRAG JSON:
```bash
# Generate full graph with HTML visualization
graphify .

# Or run with directed relationships preserved:
graphify . --directed
```

### 2. Querying Concepts Across the Codebase
```bash
# Shortest path between calibration and measurement
graphify path "CameraCalibration" "MetricMeasurement"

# Explain a specific component and its neighborhood
graphify explain "inference/measure.py"
```

### 3. Generated Graph Artifacts
When graphify runs, it populates `graphify-out/` with:
- `graph.html`: Interactive 2D/3D physics-based network visualization.
- `graph.json`: GraphRAG-ready node and edge list with community clusters.
- `GRAPH_REPORT.md`: Comprehensive plain-language structural audit.
