"""Compile ideal target states into optimized ECD circuit parameters."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

from metriq_qudits.benchmark.circuit_io import save_circuits
from metriq_qudits.parallel import parallel_map
from metriq_qudits.system_config import SystemConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILED_DIR = REPO_ROOT / ".cache" / "compiled_circuits"
CALIBRATION_DIR = REPO_ROOT / ".cache" / "calibration"

N_UNITARIES = 25
SEED = 42
DEFAULT_N_JOBS = int(os.environ.get("N_JOBS", "1"))

MIN_BUFFERS = 3
MAX_BUFFERS = 35
N_CALIBRATION_CIRCUITS = 5
PENALTY_WEIGHT = 0.1
STABILITY_THRESHOLD = 1e-2
N_TEST_EXTRA = tuple(range(1, 13))
K_STEP = 1
K_MAX_FACTOR = 3
ERROR_THRESHOLD = 0.01
BATCH_SIZE = 256
OPTIMIZATION_STEPS = 2000
CHECK_EVERY = 100
RESAMPLE_CAP = 2


def parameter_counting_floor(config: SystemConfig) -> int:
    """Lowest depth with enough real parameters to represent a target state."""
    return max(1, int(np.ceil((config.dimension - 2) / 2)))


def heuristic_depth(config: SystemConfig) -> int:
    """Known-comfortable depth used for Fock-buffer calibration."""
    return max(4, int(np.ceil(np.sqrt(config.dimension))) * 4)


def sample_target(config: SystemConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw ``|psi> = U|0>`` from the Haar ensemble."""
    return unitary_group.rvs(config.dimension, random_state=rng)[:, 0]


def compiled_path(
    config: SystemConfig,
    depth: int,
    n_unitaries: int = N_UNITARIES,
    seed: int = SEED,
) -> str:
    filename = f"{config.key}_k{depth}_nu{n_unitaries}_seed{seed}.npz"
    return str(COMPILED_DIR / filename)


def probe_path(config: SystemConfig) -> str:
    return str(CALIBRATION_DIR / f"probe_{config.key}.npz")


def _buffer_count_to_penalize(num_buffers: int) -> int:
    return max(1, num_buffers // 2)


def _make_job(
    target,
    config: SystemConfig,
    k_init,
    k_max,
    n_cavity,
    n_penalize,
    seed,
    stability_threshold=None,
):
    # Deferred import keeps JAX (pulled in by ecd_parameter_finder) out of this
    # module's load path, preserving the spawn-worker startup cost.
    from metriq_qudits.compilation.ecd_parameter_finder import CompileJob, OptimizerConfig

    opt = OptimizerConfig(
        n_penalize=n_penalize,
        penalty_weight=PENALTY_WEIGHT,
        err_th=ERROR_THRESHOLD,
        batch_size=BATCH_SIZE,
        n_steps=OPTIMIZATION_STEPS,
        check_every=CHECK_EVERY,
        stability_th=stability_threshold,
        n_test_extra=N_TEST_EXTRA,
    )
    return CompileJob(
        d=config.d, num_modes=1, config=opt, target=target,
        k_init=k_init, k_max=k_max, k_step=K_STEP, N=n_cavity, seed=seed,
    )


def _calibrate_buffers(
    config: SystemConfig,
    calibration_depth: int,
    rng: np.random.Generator,
):
    """Find a Fock truncation whose compiled circuits are stable."""
    from metriq_qudits.compilation.ecd_parameter_finder import (
        compile_circuit_worker,
        stability_infidelity,
    )

    print(
        "\n  [compile] calibrating Fock buffers ...  "
        f"{config.key} k_init={calibration_depth}"
    )
    targets = [sample_target(config, rng) for _ in range(N_CALIBRATION_CIRCUITS)]
    seed = int(rng.integers(0, 2**31))

    best_buffers = best_n_penalize = None
    tried = []
    max_stabilities = []
    stable_flags = []
    all_curves = []

    for num_buffers in range(MIN_BUFFERS, MAX_BUFFERS + 1):
        n_penalize = _buffer_count_to_penalize(num_buffers)
        n_cavity = config.d + num_buffers
        jobs = [
            _make_job(
                targets[i], config, calibration_depth, calibration_depth,
                n_cavity, n_penalize, seed + i,
            )
            for i in range(N_CALIBRATION_CIRCUITS)
        ]
        compiled = [compile_circuit_worker(job) for job in jobs]

        curves = np.full((N_CALIBRATION_CIRCUITS, len(N_TEST_EXTRA)), np.nan)
        if any(circuit is not None for circuit in compiled):
            for i, circuit in enumerate(compiled):
                if circuit is not None:
                    curves[i] = stability_infidelity(
                        circuit,
                        targets[i],
                        [n_cavity + extra for extra in N_TEST_EXTRA],
                        config.d,
                        1,
                    )
            max_stability = float(np.nanmax(curves))
            stable = max_stability < STABILITY_THRESHOLD
            marker = "STABLE <" if stable else ""
            print(
                f"    buffers={num_buffers}  N={n_cavity}  "
                f"max_stab={max_stability:.2e}  {marker}"
            )
        else:
            max_stability, stable = np.nan, False
            print(f"    buffers={num_buffers}  N={n_cavity}  all circuits failed")

        tried.append((num_buffers, n_penalize))
        max_stabilities.append(max_stability)
        stable_flags.append(stable)
        all_curves.append(curves)

        if stable:
            best_buffers, best_n_penalize = num_buffers, n_penalize
            break

    if best_buffers is None:
        best_buffers = MAX_BUFFERS
        best_n_penalize = _buffer_count_to_penalize(MAX_BUFFERS)
        print(f"  WARNING: no stable config found. Using num_buffers={best_buffers}")
    else:
        print(
            f"  -> calibrated: num_buffers={best_buffers}  "
            f"n_pen={best_n_penalize}"
        )

    tried = np.asarray(tried, dtype=np.int32)
    path = CALIBRATION_DIR / f"cal_{config.key}_k{calibration_depth}.npz"
    np.savez(
        path,
        num_buffers_tried=tried[:, 0],
        n_pen_tried=tried[:, 1],
        max_stab=np.asarray(max_stabilities),
        stable=np.asarray(stable_flags),
        curves=np.asarray(all_curves),
        N_test_extra=np.asarray(N_TEST_EXTRA),
        best_buffers=best_buffers,
        best_n_pen=best_n_penalize,
        d=config.d,
        num_modes=1,
        k=calibration_depth,
        penalty_weight=PENALTY_WEIGHT,
        stability_th=STABILITY_THRESHOLD,
        seed=SEED,
    )
    print(f"  Saved calibration -> {path}")
    return best_buffers, best_n_penalize, targets


def _probe_minimum_depth(
    targets,
    config: SystemConfig,
    depth_floor: int,
    calibration_depth: int,
    n_cavity: int,
    n_penalize: int,
    rng: np.random.Generator,
    n_jobs: int,
) -> int:
    """Probe calibration targets so production avoids repeatedly hard depths."""
    from metriq_qudits.compilation.ecd_parameter_finder import compile_circuit_worker

    print(
        "\n  [compile] depth probe ...  "
        f"k_floor={depth_floor}  k_cal={calibration_depth}  N={n_cavity}"
    )
    jobs = [
        _make_job(
            target,
            config,
            depth_floor,
            calibration_depth,
            n_cavity,
            n_penalize,
            int(rng.integers(0, 2**31)),
            stability_threshold=STABILITY_THRESHOLD,
        )
        for target in targets
    ]
    probed = [
        circuit
        for circuit in parallel_map(compile_circuit_worker, jobs, n_jobs)
        if circuit is not None
    ]
    if not probed:
        print(
            "  WARNING: no probe target accepted <= k_cal. "
            f"production sweeps from k_floor={depth_floor}"
        )
        return depth_floor

    path = probe_path(config)
    save_circuits(
        probed,
        {
            "d": config.d,
            "k": depth_floor,
            "num_modes": 1,
            "ansatz": "haar",
            "N_cav": n_cavity,
            "N_unitaries": len(probed),
            "seed": SEED,
            "trace_stride": CHECK_EVERY,
            "n_penalize": n_penalize,
            "penalty_weight": PENALTY_WEIGHT,
        },
        path,
    )
    depths = sorted(circuit.depth for circuit in probed)
    production_start = max(depth_floor, depths[0] - 1)
    print(
        f"  -> probe depths {depths}  -> production k_start={production_start}  "
        f"({len(probed)}/{len(targets)} accepted, saved -> {path})"
    )
    return production_start


def compile_circuits(
    config: SystemConfig,
    *,
    overwrite: bool = False,
    n_jobs: int = DEFAULT_N_JOBS,
) -> str:
    """Compile a reproducible ensemble and return its versioned cache path."""
    from metriq_qudits.compilation.ecd_parameter_finder import compile_circuit_worker

    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    if config.dimension != config.d:
        raise ValueError("expected a single-qudit system")

    depth_floor = parameter_counting_floor(config)
    calibration_depth = heuristic_depth(config)
    output_path = compiled_path(config, depth_floor)
    if os.path.exists(output_path) and not overwrite:
        print(f"  [compile] cached -> {os.path.basename(output_path)}")
        return output_path

    calibration_rng = np.random.default_rng([SEED, config.d, 1, 0])
    production_rng = np.random.default_rng([SEED, config.d, 1, 1])
    probe_rng = np.random.default_rng([SEED, config.d, 1, 2])

    num_buffers, n_penalize, calibration_targets = _calibrate_buffers(
        config, calibration_depth, calibration_rng,
    )
    n_cavity = config.d + num_buffers
    production_start = _probe_minimum_depth(
        calibration_targets,
        config,
        depth_floor,
        calibration_depth,
        n_cavity,
        n_penalize,
        probe_rng,
        n_jobs,
    )
    maximum_depth = calibration_depth * K_MAX_FACTOR
    print(
        "\n  [compile] production  "
        f"N_cav={n_cavity}  k_start={production_start}  k_max={maximum_depth}  "
        f"N_unitaries={N_UNITARIES}  batch={BATCH_SIZE}  N_JOBS={n_jobs}"
    )

    max_attempts = RESAMPLE_CAP * N_UNITARIES
    compiled = []
    n_attempted = 0
    while len(compiled) < N_UNITARIES and n_attempted < max_attempts:
        n_new = min(N_UNITARIES - len(compiled), max_attempts - n_attempted)
        jobs = [
            _make_job(
                sample_target(config, production_rng),
                config,
                production_start,
                maximum_depth,
                n_cavity,
                n_penalize,
                int(production_rng.integers(0, 2**31)),
                stability_threshold=STABILITY_THRESHOLD,
            )
            for _ in range(n_new)
        ]
        for result in parallel_map(compile_circuit_worker, jobs, n_jobs):
            n_attempted += 1
            if result is None:
                print(
                    f"    [attempt {n_attempted}] FAILED "
                    "(no stable candidate <= err_th)"
                )
                continue
            compiled.append(result)
            print(
                f"    [attempt {n_attempted}] done  err={result.infidelity:.2e}  "
                f"k={result.depth}  leakage={result.boundary_leakage:.1e}  "
                f"({len(compiled)}/{N_UNITARIES})"
            )

    if len(compiled) < N_UNITARIES:
        print(
            f"  WARNING: only {len(compiled)}/{N_UNITARIES} circuits "
            f"after {n_attempted} attempts"
        )
    save_circuits(
        compiled,
        {
            "d": config.d,
            "k": depth_floor,
            "k_start": production_start,
            "num_modes": 1,
            "ansatz": "haar",
            "N_cav": n_cavity,
            "N_unitaries": len(compiled),
            "n_attempted": n_attempted,
            "seed": SEED,
            "num_buffers": num_buffers,
            "trace_stride": CHECK_EVERY,
            "n_penalize": n_penalize,
            "penalty_weight": PENALTY_WEIGHT,
        },
        output_path,
    )
    print(f"  Saved {len(compiled)} circuits -> {output_path}")
    return output_path
