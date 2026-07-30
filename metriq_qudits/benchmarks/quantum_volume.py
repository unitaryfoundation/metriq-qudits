"""Quantum-volume benchmark runner.
"""

from __future__ import annotations

import os
import platform

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from metriq_qudits.compilation.compile import compile_circuits
from metriq_qudits.pulses.build import build_circuit_pulses
from metriq_qudits.simulation.sweep import run_noise_sweep, run_noiseless
from metriq_qudits.system_config import SystemConfig

N_JOBS = int(os.environ.get("N_JOBS", "1"))

CONFIGS = tuple(SystemConfig(d=d) for d in (4, 6, 8, 10, 12, 14, 16))
CONFIG_BY_KEY = {config.key: config for config in CONFIGS}


def run_experiment(
    config: SystemConfig,
    *,
    correct_phases: bool = True,
    backend: str = "dynamiqs",
    include_noise_sweep: bool = True,
    overwrite: bool = False,
    n_jobs: int = N_JOBS,
    optimizer: str = "lbfgs",
    use_probe: bool = False,
    t1_values: np.ndarray | None = None,
    t2_values: np.ndarray | None = None,
) -> None:
    """Execute compilation, pulse construction, and simulation."""
    compiled_path = compile_circuits(
        config, overwrite=overwrite, n_jobs=n_jobs, optimizer=optimizer,
        use_probe=use_probe,
    )

    pulse_path = build_circuit_pulses(
        compiled_path,
        correct_phases=correct_phases,
        overwrite=overwrite,
        n_jobs=n_jobs,
    )

    run_noiseless(
        pulse_path,
        compiled_path,
        correct_phases=correct_phases,
        backend=backend,
        diagnostics_dir=None,
        overwrite=overwrite,
        n_jobs=n_jobs,
        t1_values=t1_values,
        t2_values=t2_values,
    )
    if include_noise_sweep:
        run_noise_sweep(
            pulse_path,
            compiled_path,
            correct_phases=correct_phases,
            backend=backend,
            overwrite=overwrite,
            n_jobs=n_jobs,
            t1_values=t1_values,
            t2_values=t2_values,
        )
