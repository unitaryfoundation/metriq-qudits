"""
benchmark-agnostic helper functions for:
    1. Compilation
    2. Pulse building
    3. Simulation
"""

from __future__ import annotations

import os
import platform

# JAX can otherwise select an unsupported backend while importing the simulator.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial

import numpy as np

from metriq_qudits.compilation.circuit_io import CompiledCircuit
from metriq_qudits.compilation.ecd_parameter_finder import (
    ECDParameterFinder,
    OptimizerConfig,
)
from metriq_qudits.parallel import parallel_map
from metriq_qudits.pulses.drive_envelopes import CavityMode, TransmonAncilla
from metriq_qudits.pulses.ecd_pulse_builder import CircuitWaveforms, ECDPulseBuilder
from metriq_qudits.simulation.displaced_frame import DisplacedFrameSimulator
from metriq_qudits.simulation.displaced_frame_dynamiqs import DisplacedFrameSimulatorDQ

N_JOBS = int(os.environ.get("N_JOBS", "1"))


def compile_circuit(unitary: np.ndarray, *, d: int, n_cavity: int, k_init: int,
                    k_max: int, optimizer_config: OptimizerConfig) -> CompiledCircuit | None:
    """Compile one target to the shallowest converging ECD circuit, or None if fails to converge."""
    finder = ECDParameterFinder(d=d, num_modes=1, config=optimizer_config)
    return finder.find_parameters_adaptive_k(
        unitary, k_init=k_init, k_max=k_max, N=n_cavity,
        rng=np.random.default_rng(),
    )


def compile_circuits(unitaries: list[np.ndarray], *, d: int, n_cavity: int,
                     k_init: int, k_max: int, optimizer_config: OptimizerConfig,
                     n_jobs: int = N_JOBS,
                     on_progress=None) -> list[CompiledCircuit | None]:
    """Parallel compilation across circuits."""
    worker = partial(compile_circuit, d=d, n_cavity=n_cavity, k_init=k_init,
                     k_max=k_max, optimizer_config=optimizer_config)
    return list(parallel_map(worker, unitaries, n_jobs, on_progress=on_progress))


# Peak intra-gate cavity displacement used when shaping ECD pulses (Eickbusch et al. 2022).
# Device physics (chi, chi_prime, self-Kerr, ancilla timing) lives in the CavityMode and
# TransmonAncilla dataclass defaults.
PEAK_DISPLACEMENT = 10.0


def build_circuit_pulse(circuit: CompiledCircuit | None, *,
                        peak_displacement: float = PEAK_DISPLACEMENT,
                        correct_phases: bool = True) -> CircuitWaveforms | None:
    """Build one circuit's physical waveforms."""
    if circuit is None:
        return None
    builder = ECDPulseBuilder([CavityMode()], TransmonAncilla())
    return builder.compile_circuit(
        betas=circuit.betas,
        rotations=circuit.rotations,
        peak_amplitude=peak_displacement,
        correct_cavity_phases=correct_phases,
    )


def build_circuit_pulses(circuits: list[CompiledCircuit | None], *,
                         peak_displacement: float = PEAK_DISPLACEMENT,
                         correct_phases: bool = True,
                         n_jobs: int = N_JOBS) -> list[CircuitWaveforms | None]:
    """Build waveforms for the whole ensemble in parallel."""
    worker = partial(build_circuit_pulse, peak_displacement=peak_displacement,
                     correct_phases=correct_phases)
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
                      n_jobs: int = N_JOBS,
                      on_progress=None) -> list[np.ndarray | None]:
    """Simulate the ensemble in parallel."""
    worker = partial(simulate_circuit, n_cavity=n_cavity, backend=backend,
                     t1_us=t1_us, t2_us=t2_us)
    return list(parallel_map(worker, pulses, n_jobs, on_progress=on_progress))


def save_result(path, result, **extra) -> None:
    """Persist a benchmark result's metric fields to an npz."""
    arrays = {key: np.asarray(value) for key, value in result.model_dump().items()}
    arrays.update({key: np.asarray(value) for key, value in extra.items()})
    np.savez(str(path), **arrays)


def load_result(path, result_cls):
    """Load a benchmark result of the given type from an npz."""
    fields = result_cls.model_fields
    with np.load(str(path)) as data:
        return result_cls(**{key: data[key].tolist()
                             for key in data.files if key in fields})
