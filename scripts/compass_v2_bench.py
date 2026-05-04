"""Thin launcher for COMPASS v2 benchmark (wrapper around ``src.compass.v2.bench.main``)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compass.v2.bench import main

if __name__ == "__main__":
    main()
