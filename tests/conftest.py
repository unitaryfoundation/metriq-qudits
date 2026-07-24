"""Shared pytest configuration.

Slow tests exercise the real JAX optimizer end to end and take minutes, so they
are opt-in. Run them with ``pytest --slow``. Without the flag they are skipped.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run slow tests (full-pipeline e2e runs of the JAX optimizer)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
