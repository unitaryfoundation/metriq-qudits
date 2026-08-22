from __future__ import annotations

import os
import platform

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

from metriq_qudits.compilation.ecd_parameter_finder import (
    CompileJob,
    OptimizerConfig,
    compile_circuit_worker,
    stability_infidelity,
)
from metriq_qudits.parallel import parallel_map

# Fock-buffer calibration search extents and optimizer policy.
MIN_BUFFERS = 3
MAX_BUFFERS = 35
N_CALIBRATION_CIRCUITS = 5
N_TEST_EXTRA = tuple(range(1, 13))
K_MAX_FACTOR = 3
PENALTY_WEIGHT = 0.1
STABILITY_THRESHOLD = 1e-2


@dataclass(frozen=True)
class Calibration:
    """Per-config Fock truncation, penalty count, and depth window."""

    n_cavity: int
    n_penalize: int
    k_start: int
    k_max: int


def parameter_counting_floor(dimension: int) -> int:
    """Lowest depth with enough real parameters to represent a target state."""
    return max(1, int(np.ceil((dimension - 2) / 2)))


def heuristic_depth(dimension: int) -> int:
    """Comfortable depth used for buffer calibration and the k_max window."""
    return max(4, int(np.ceil(np.sqrt(dimension))) * 4)


def _buffer_count_to_penalize(num_buffers: int) -> int:
    return max(1, num_buffers // 2)


def compile_config(cal: Calibration, *, stability_th: float = STABILITY_THRESHOLD,
                   penalty_weight: float = PENALTY_WEIGHT) -> OptimizerConfig:
    """Production optimizer config for compiling against a calibration."""
    return OptimizerConfig(n_penalize=cal.n_penalize, penalty_weight=penalty_weight,
                           stability_th=stability_th, n_test_extra=N_TEST_EXTRA)


def _buffer_diagnostics(d, depth, stability_th, tried, max_stab, curves, best):
    return {
        "d": d,
        "k": depth,
        "stability_th": stability_th,
        "N_test_extra": np.array(N_TEST_EXTRA),
        "num_buffers_tried": np.array(tried),
        "max_stab": np.array(max_stab),
        "curves": np.array(curves),
        "best_buffers": best,
    }


def _calibrate_buffers(d, depth, targets, seed, penalty_weight, stability_th, n_jobs):
    """Smallest Fock buffer whose calibration circuits stay stable off the boundary."""
    tried, max_stab, curves = [], [], []
    for num_buffers in range(MIN_BUFFERS, MAX_BUFFERS + 1):
        n_penalize = _buffer_count_to_penalize(num_buffers)
        n_cavity = d + num_buffers
        # stability_th=None: accept the best converged candidate, then measure its
        # stability at N + n_test_extra ourselves below.
        cfg = OptimizerConfig(n_penalize=n_penalize, penalty_weight=penalty_weight,
                              stability_th=None, n_test_extra=N_TEST_EXTRA)
        jobs = [CompileJob(d=d, num_modes=1, config=cfg, target=targets[i],
                           k_init=depth, k_max=depth, k_step=1, N=n_cavity, seed=seed + i)
                for i in range(len(targets))]
        compiled = list(parallel_map(compile_circuit_worker, jobs, n_jobs))
        circuit_curves = np.full((len(targets), len(N_TEST_EXTRA)), np.nan)
        for i, circuit in enumerate(compiled):
            if circuit is not None:
                circuit_curves[i] = stability_infidelity(
                    circuit, targets[i], [n_cavity + e for e in N_TEST_EXTRA], d, 1)
        valid = circuit_curves[~np.all(np.isnan(circuit_curves), axis=1)]
        peak = float(np.nanmax(valid)) if valid.size else np.nan
        tried.append(num_buffers)
        max_stab.append(peak)
        curves.append(circuit_curves)
        if valid.size and peak < stability_th:
            return num_buffers, n_penalize, _buffer_diagnostics(
                d, depth, stability_th, tried, max_stab, curves, num_buffers)
    return MAX_BUFFERS, _buffer_count_to_penalize(MAX_BUFFERS), _buffer_diagnostics(
        d, depth, stability_th, tried, max_stab, curves, MAX_BUFFERS)


def _probe_start_depth(d, floor, depth, n_cavity, n_penalize, targets, seed,
                       penalty_weight, stability_th, n_jobs):
    """Compile the calibration targets bottom-up to pick a shallow production start."""
    cfg = OptimizerConfig(n_penalize=n_penalize, penalty_weight=penalty_weight,
                          stability_th=stability_th, n_test_extra=N_TEST_EXTRA)
    jobs = [CompileJob(d=d, num_modes=1, config=cfg, target=target,
                       k_init=floor, k_max=depth, k_step=1, N=n_cavity, seed=seed + i)
            for i, target in enumerate(targets)]
    probed = [c for c in parallel_map(compile_circuit_worker, jobs, n_jobs) if c is not None]
    if not probed:
        return floor
    return max(floor, min(circuit.depth for circuit in probed) - 1)


def calibrate(d: int, *, n_jobs: int, penalty_weight: float = PENALTY_WEIGHT,
              stability_th: float = STABILITY_THRESHOLD) -> Calibration:
    """Fit the Fock truncation, penalty count, and depth window for one dimension."""
    dimension = d
    floor = parameter_counting_floor(dimension)
    depth = heuristic_depth(dimension)
    rng = np.random.default_rng([d, 1])
    targets = [unitary_group.rvs(dimension, random_state=rng)[:, 0]
               for _ in range(N_CALIBRATION_CIRCUITS)]
    buffer_seed = int(rng.integers(0, 2**31))
    probe_seed = int(rng.integers(0, 2**31))
    num_buffers, n_penalize, diagnostics = _calibrate_buffers(
        d, depth, targets, buffer_seed, penalty_weight, stability_th, n_jobs)
    n_cavity = d + num_buffers
    k_start = _probe_start_depth(
        d, floor, depth, n_cavity, n_penalize, targets, probe_seed,
        penalty_weight, stability_th, n_jobs)
    cal = Calibration(n_cavity=n_cavity, n_penalize=n_penalize,
                      k_start=k_start, k_max=K_MAX_FACTOR * depth)
    return cal, diagnostics


def _write(path, cal: Calibration, diagnostics=None) -> None:
    arrays = dict(n_cavity=cal.n_cavity, n_penalize=cal.n_penalize,
                  k_start=cal.k_start, k_max=cal.k_max)
    if diagnostics:
        arrays.update(diagnostics)
    np.savez(str(path), **arrays)


def _read(path) -> Calibration:
    with np.load(str(path)) as data:
        return Calibration(
            n_cavity=int(data["n_cavity"]), n_penalize=int(data["n_penalize"]),
            k_start=int(data["k_start"]), k_max=int(data["k_max"]),
        )


def load_or_calibrate(path, d: int, *, overwrite: bool = False, n_jobs: int,
                      penalty_weight: float = PENALTY_WEIGHT,
                      stability_th: float = STABILITY_THRESHOLD) -> Calibration:
    """Existence-cached calibration for one dimension."""
    path = Path(path)
    if not overwrite and path.exists():
        return _read(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cal, diagnostics = calibrate(d, n_jobs=n_jobs, penalty_weight=penalty_weight,
                                 stability_th=stability_th)
    _write(path, cal, diagnostics)
    return cal
