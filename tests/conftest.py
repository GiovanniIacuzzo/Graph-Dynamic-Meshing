"""Stable import paths for the Python source tree and local training utilities."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
