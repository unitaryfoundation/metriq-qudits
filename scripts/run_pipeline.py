"""Compatibility wrapper for the installed ``metriq-qudits`` command."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metriq_qudits.cli import (
    CONFIGS,
    CONFIG_BY_KEY,
    N_JOBS,
    _parse_args,
    main,
    run_experiment,
)

__all__ = [
    "CONFIGS",
    "CONFIG_BY_KEY",
    "N_JOBS",
    "_parse_args",
    "main",
    "run_experiment",
]


if __name__ == "__main__":
    main()
