"""Compatibility wrapper for :mod:`metriq_qudits.plot_results`."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metriq_qudits.plot_results import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
