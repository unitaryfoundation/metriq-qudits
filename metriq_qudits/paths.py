"""Configurable filesystem locations for generated pipeline artifacts."""

from __future__ import annotations

import os
from pathlib import Path

OUTPUT_DIR_ENV = "METRIQ_QUDITS_OUTPUT_DIR"


def set_output_dir(path: str | os.PathLike[str]) -> Path:
    """Set and return the output root used by this process and its workers."""
    resolved = Path(path).expanduser().resolve()
    os.environ[OUTPUT_DIR_ENV] = str(resolved)
    return resolved


def output_dir() -> Path:
    """Return the artifact root, defaulting to ``outputs/`` in the current directory."""
    configured = os.environ.get(OUTPUT_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / "outputs").resolve()


def data_dir(*parts: str) -> Path:
    """Return a path beneath the generated-artifact root."""
    return output_dir() / Path(*parts)


def plots_dir(*parts: str) -> Path:
    """Return a path beneath the generated-plots directory."""
    return output_dir() / "plots" / Path(*parts)
