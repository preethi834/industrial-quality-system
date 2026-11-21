#!/usr/bin/env python
"""Factory simulator CLI. Usage: uv run python scripts/simulate.py --help"""

from __future__ import annotations

import sys
from pathlib import Path

from iqs_sim.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sys.exit(main(default_taxonomy=REPO_ROOT / "configs" / "taxonomy" / "causes.yaml"))
