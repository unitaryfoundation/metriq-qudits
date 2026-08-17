"""
Quantum-volume benchmark runner.
"""

from __future__ import annotations

import os
import platform

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy.stats import unitary_group

from metriq_qudits.benchmarks.helpers import (
    N_JOBS,
    build_circuit_pulses,
    compile_circuits,
    save_scores,
    simulate_circuits,
)
from metriq_qudits.compilation.circuit_io import load_circuits, save_circuits
from metriq_qudits.compilation.ecd_parameter_finder import OptimizerConfig
from metriq_qudits.metrics import eval_circuit
from metriq_qudits.paths import data_dir
from metriq_qudits.pulses.pulse_io import load_pulses, save_pulses
from metriq_qudits.system_config import SystemConfig

BENCHMARK_NAME = "quantum_volume"

# Ancilla coherence grid for the T1/T2 noise sweep, in microseconds.
T1_US = (5.0, 50.0, 100.0)
T2_US = (10.0, 100.0, 200.0)

CONFIGS = tuple(SystemConfig(d=d) for d in (4, 6, 8, 10, 12, 14, 16))
CONFIG_BY_KEY = {config.key: config for config in CONFIGS}


def run_dir(config: SystemConfig) -> Path:
    return data_dir("runs", BENCHMARK_NAME, config.key)


def sample_targets(config: SystemConfig, n_unitaries: int, seed: int) -> list[np.ndarray]:
    """
    Draw the ensemble of targets |psi> = U|0> from the Haar measure.
    """
    rng = np.random.default_rng([seed, config.d])
    return [unitary_group.rvs(config.dimension, random_state=rng)[:, 0]
            for _ in range(n_unitaries)]


def run_qv_experiment(
    config: SystemConfig,
    params: BaseModel,
    *,
    correct_phases: bool = True,
    backend: str = "dynamiqs",
    include_noise_sweep: bool = True,
    overwrite: bool = False,
    n_jobs: int = N_JOBS,
    optimizer: str = "lbfgs",
    t1_values: np.ndarray | None = None,
    t2_values: np.ndarray | None = None,
) -> None:
    """Execute compilation, pulse construction, and simulation."""
    n_unitaries = params.n_unitaries
    seed = params.seed
    t1_values = T1_US if t1_values is None else t1_values
    t2_values = T2_US if t2_values is None else t2_values
    circuits_dir = run_dir(config) / "circuits"
    pulses_dir = run_dir(config) / "pulses"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    pulses_dir.mkdir(parents=True, exist_ok=True)
    circuit_paths = [circuits_dir / f"{i:04d}.npz" for i in range(n_unitaries)]
    pulse_paths = [pulses_dir / f"{i:04d}.npz" for i in range(n_unitaries)]

    # Compile stage: compile the ensemble, then save every circuit.
    # Overwrite gate: run when forced or missing, otherwise reuse what is on disk.
    if overwrite or not all(path.exists() for path in circuit_paths):
        unitaries = sample_targets(config, n_unitaries, seed)
        # Provisional truncation and depth window (calibration will set these later).
        circuits = compile_circuits(
            unitaries, d=config.d, n_cavity=config.d + 4, k_init=1,
            k_max=4 * config.d, optimizer_config=OptimizerConfig(), n_jobs=n_jobs,
        )
        save_circuits(circuit_paths, circuits)
    circuits = load_circuits(circuit_paths)

    # Pulse stage: build waveforms from the circuits, then save every pulse.
    compiled = [i for i, circuit in enumerate(circuits) if circuit is not None]
    if overwrite or not all(pulse_paths[i].exists() for i in compiled):
        pulses = build_circuit_pulses(circuits, correct_phases=correct_phases, n_jobs=n_jobs)
        save_pulses(pulse_paths, pulses)

    # Simulation stage (noiseless): simulate each pulse, then score the states separately.
    metrics_dir = run_dir(config) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    noiseless_path = metrics_dir / "noiseless.npz"
    if overwrite or not noiseless_path.exists():
        pulses = load_pulses(pulse_paths)
        n_cavity = config.d + 4  # provisional truncation (matches compile_circuit)
        states = simulate_circuits(pulses, n_cavity=n_cavity, backend=backend, n_jobs=n_jobs)
        metrics = [
            eval_circuit(state, circuit.target_state, config.d, 1, n_cavity)
            if state is not None else None
            for state, circuit in zip(states, circuits)
        ]
        save_scores(noiseless_path, metrics)
    with np.load(str(noiseless_path)) as summary:
        print(f"  [noiseless] {config.key}  HOG={float(summary['hog_mean']):.4f}  "
              f"XEB={float(summary['xeb_mean']):.4f}  F={float(summary['fid_mean']):.4f}")

    # Noise sweep: one aggregated file per (T1, T2) grid point, keeping T2 <= 2 T1.
    if include_noise_sweep:
        sweep_dir = metrics_dir / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        n_cavity = config.d + 4  # provisional truncation (matches compile_circuit)
        pulses = load_pulses(pulse_paths)
        for i1, t1_us in enumerate(t1_values):
            for i2, t2_us in enumerate(t2_values):
                if t2_us > 2 * t1_us:
                    continue
                point_path = sweep_dir / f"{i1:02d}_{i2:02d}.npz"
                if not overwrite and point_path.exists():
                    continue
                states = simulate_circuits(
                    pulses, n_cavity=n_cavity, backend=backend,
                    t1_us=t1_us, t2_us=t2_us, n_jobs=n_jobs,
                )
                metrics = [
                    eval_circuit(state, circuit.target_state, config.d, 1, n_cavity)
                    if state is not None else None
                    for state, circuit in zip(states, circuits)
                ]
                save_scores(point_path, metrics)
                with np.load(str(point_path)) as summary:
                    print(f"  [sweep] T1={t1_us:.0f}us T2={t2_us:.0f}us  "
                          f"HOG={float(summary['hog_mean']):.3f}  "
                          f"XEB={float(summary['xeb_mean']):.3f}  "
                          f"F={float(summary['fid_mean']):.3f}")