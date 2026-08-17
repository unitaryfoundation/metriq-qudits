"""
Quantum-volume benchmark runner.
"""

from __future__ import annotations

import os
import platform

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

from metriq_qudits.compilation.circuit_io import (
    CompiledCircuit,
    load_circuits,
    save_circuits,
)
from metriq_qudits.compilation.ecd_parameter_finder import (
    ECDParameterFinder,
    OptimizerConfig,
)
from metriq_qudits.metrics import eval_circuit
from metriq_qudits.parallel import parallel_map
from metriq_qudits.paths import data_dir
from metriq_qudits.pulses.drive_envelopes import CavityMode, TransmonAncilla
from metriq_qudits.pulses.ecd_pulse_builder import CircuitWaveforms, ECDPulseBuilder
from metriq_qudits.pulses.pulse_io import load_pulses, save_pulses
from metriq_qudits.simulation.displaced_frame import DisplacedFrameSimulator
from metriq_qudits.simulation.displaced_frame_dynamiqs import DisplacedFrameSimulatorDQ
from metriq_qudits.system_config import SystemConfig

BENCHMARK_NAME = "quantum_volume"
N_JOBS = int(os.environ.get("N_JOBS", "1"))

# Peak intra-gate cavity displacement used when shaping ECD pulses (Eickbusch et al. 2022).
# Device physics (chi, chi_prime, self-Kerr, ancilla timing) lives in the CavityMode and
# TransmonAncilla dataclass defaults.
PEAK_TRANSIENT_DISPLACEMENT = 10

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


def compile_circuit(config: SystemConfig, unitary: np.ndarray) -> CompiledCircuit | None:
    """
    Returns the CompiledCircuit at the shallowest depth that converges, or None if
    no depth in the window produces a stable circuit.
    """
    # Provisional truncation and depth window (calibration will set these later).
    n_cavity = config.d + 4
    finder = ECDParameterFinder(d=config.d, num_modes=1, config=OptimizerConfig())
    circuit = finder.find_parameters_adaptive_k(
        unitary, k_init=1, k_max=4 * config.d, N=n_cavity,
        rng=np.random.default_rng(),
    )
    return circuit

def compile_circuits(
    config: SystemConfig, unitaries: list[np.ndarray], *, n_jobs: int = N_JOBS,
) -> list[CompiledCircuit | None]:
    return list(parallel_map(partial(compile_circuit, config), unitaries, n_jobs))


def build_circuit_pulse(circuit: CompiledCircuit | None, *,
                        correct_phases: bool = True) -> CircuitWaveforms | None:
    """Build one circuit's physical waveforms. Pure: object in, object out."""
    if circuit is None:
        return None
    builder = ECDPulseBuilder([CavityMode()], TransmonAncilla())
    return builder.compile_circuit(
        betas=circuit.betas,
        rotations=circuit.rotations,
        peak_amplitude=PEAK_TRANSIENT_DISPLACEMENT,
        correct_cavity_phases=correct_phases,
    )


def build_circuit_pulses(
    circuits: list[CompiledCircuit | None], *,
    correct_phases: bool = True, n_jobs: int = N_JOBS,
) -> list[CircuitWaveforms | None]:
    """Build waveforms for the whole ensemble in parallel. Order preserved."""
    worker = partial(build_circuit_pulse, correct_phases=correct_phases)
    return list(parallel_map(worker, circuits, n_jobs))


def make_simulator(backend: str, n_cavity: int):
    """Construct the displaced-frame simulator for one backend."""
    if backend == "dynamiqs":
        return DisplacedFrameSimulatorDQ(cavity_dim=n_cavity, mode=CavityMode())
    return DisplacedFrameSimulator(cavity_dim=n_cavity, mode=CavityMode())


def simulate_circuit(pulse: CircuitWaveforms | None, *, n_cavity: int,
                     backend: str = "dynamiqs",
                     t1_us: float | None = None,
                     t2_us: float | None = None) -> np.ndarray | None:
    """Simulate one pulse and return the physical cavity density matrix.

    Pure: object in, state out. Scoring the state against a target is separate."""
    if pulse is None:
        return None
    simulator = make_simulator(backend, n_cavity)
    result, alpha = simulator.simulate(
        epsilon=pulse.cavity_drives[0],
        omega=pulse.ancilla_drive,
        T1_us=t1_us,
        T2_us=t2_us,
    )
    final_state = simulator.to_physical_frame(
        result.states[-1], alpha, cavity_phase=pulse.final_cavity_phases[0],
    )
    return final_state.ptrace([0]).full()


def simulate_circuits(pulses, *, n_cavity: int, backend: str = "dynamiqs",
                      t1_us: float | None = None, t2_us: float | None = None,
                      n_jobs: int = N_JOBS) -> list[np.ndarray | None]:
    """Simulate the ensemble in parallel. Order preserved (None where a pulse is missing)."""
    worker = partial(simulate_circuit, n_cavity=n_cavity, backend=backend,
                     t1_us=t1_us, t2_us=t2_us)
    return list(parallel_map(worker, pulses, n_jobs))


def save_scores(path, metrics) -> None:
    """Aggregate per-circuit (hog, xeb, fid) scores into a summary npz."""
    scored = np.array([m for m in metrics if m is not None], dtype=float)
    if scored.size == 0:
        raise ValueError("no circuits were scored")
    hog, xeb, fid = scored[:, 0], scored[:, 1], scored[:, 2]
    np.savez(
        str(path),
        hog=hog, xeb=xeb, fid=fid,
        hog_mean=hog.mean(), xeb_mean=xeb.mean(), fid_mean=fid.mean(),
    )


def run_qv_experiment(
    config: SystemConfig,
    *,
    correct_phases: bool = True,
    backend: str = "dynamiqs",
    include_noise_sweep: bool = True,
    overwrite: bool = False,
    n_jobs: int = N_JOBS,
    optimizer: str = "lbfgs",
    n_unitaries: int = 25,
    seed: int = 42,
    t1_values: np.ndarray | None = None,
    t2_values: np.ndarray | None = None,
) -> None:
    """Execute compilation, pulse construction, and simulation."""
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
        circuits = compile_circuits(config, unitaries, n_jobs=n_jobs)
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