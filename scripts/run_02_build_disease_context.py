#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lung_pipeline.cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--stage", "build_disease_context", *sys.argv[1:]]
    main()

