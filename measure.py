#!/usr/bin/env python3
"""Measure-It: Root CLI Entrypoint.

Forwards directly to `inference.measure.main()`.
Supports:
    python measure.py --image <path> [--output <path>] [--no_undistort] [--json]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.measure import main

if __name__ == "__main__":
    main()
