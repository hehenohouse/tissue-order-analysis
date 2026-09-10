#!/usr/bin/env python3
"""Compatibility wrapper for the Blender Fuse ``piv-frames`` command."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from blender_fuse.cli import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main(["piv-frames", *sys.argv[1:]]))
