"""Compile ideal target states into optimized ECD circuit parameters."""

from __future__ import annotations

import os

import numpy as np
from scipy.stats import unitary_group

from metriq_qudits.compilation.circuit_io import load_circuits, save_circuits
from metriq_qudits.parallel import parallel_map
from metriq_qudits.paths import data_dir
from metriq_qudits.system_config import SystemConfig

# User can modify these parameters for their purposes (e.g. lower N_UNITARIES to reduce total compilation time)
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
    return str(data_dir("compiled_circuits") / filename)


def probe_path(config: SystemConfig) -> str:
    return str(data_dir("calibration") / f"probe_{config.key}.npz")


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
    optimizer="lbfgs",
):
    # Deferred import keeps JAX (pulled in by ecd_parameter_finder) out of this
    # module's load path, preserving the spawn-worker startup cost.
    from metriq_qudits.compilation.ecd_parameter_finder import CompileJob, OptimizerConfig

    opt = OptimizerConfig(
        optimizer=optimizer,
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


def calibration_path(config: SystemConfig, calibration_depth: int) -> str:
    return str(data_dir("calibration") / f"cal_{config.key}_k{calibration_depth}.npz")


def _load_calibration(config: SystemConfig, calibration_depth: int):
    """Return (best_buffers, best_n_penalize) from a cached calibration, or None.

    Only reuse the cache when the hyperparameters that determine the result still
    match, so tweaking a constant transparently forces a recalibration."""
    path = calibration_path(config, calibration_depth)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if (
        int(data["d"]) != config.d
        or float(data["penalty_weight"]) != PENALTY_WEIGHT
        or float(data["stability_th"]) != STABILITY_THRESHOLD
    ):
        return None
    return int(data["best_buffers"]), int(data["best_n_pen"])


def _load_probe(
    config: SystemConfig,
    depth_floor: int,
    n_cavity: int,
    n_penalize: int,
    optimizer: str,
):
    """Return the cached production start depth from a saved probe, or None.

    Reuse only when the inputs that determine the probe still match, so a change to
    the calibration output, penalty, stability target, or optimizer forces a re-probe."""
    path = probe_path(config)
    if not os.path.exists(path):
        return None
    circuits, meta = load_circuits(path)
    if not circuits:
        return None
    if (
        int(meta.get("d", -1)) != config.d
        or int(meta.get("N_cav", -1)) != n_cavity
        or int(meta.get("n_penalize", -1)) != n_penalize
        or float(meta.get("penalty_weight", -1.0)) != PENALTY_WEIGHT
        or float(meta.get("stability_th", -1.0)) != STABILITY_THRESHOLD
        or str(meta.get("optimizer", "")) != optimizer
    ):
        return None
    depths = sorted(circuit.depth for circuit in circuits)
    return max(depth_floor, depths[0] - 1)


def _calibrate_buffers(
    config: SystemConfig,
    calibration_depth: int,
    rng: np.random.Generator,
    n_jobs: int = DEFAULT_N_JOBS,
    optimizer: str = "lbfgs",
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
                n_cavity, n_penalize, seed + i, optimizer=optimizer,
            )
            for i in range(N_CALIBRATION_CIRCUITS)
        ]
        compiled = list(parallel_map(compile_circuit_worker, jobs, n_jobs))

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
    path = calibration_path(config, calibration_depth)
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
    optimizer: str = "lbfgs",
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
            optimizer=optimizer,
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
            "stability_th": STABILITY_THRESHOLD,
            "optimizer": optimizer,
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
    optimizer: str = "lbfgs",
    use_probe: bool = False,
) -> str:
    """Compile a reproducible ensemble and return its versioned cache path."""
    from metriq_qudits.compilation.ecd_parameter_finder import compile_circuit_worker

    data_dir("compiled_circuits").mkdir(parents=True, exist_ok=True)
    data_dir("calibration").mkdir(parents=True, exist_ok=True)
    if config.dimension != config.d:
        raise ValueError("expected a single-mode system")

    depth_floor = parameter_counting_floor(config)
    calibration_depth = heuristic_depth(config)

    calibration_rng = np.random.default_rng([SEED, config.d, 1, 0])
    production_rng = np.random.default_rng([SEED, config.d, 1, 1])
    probe_rng = np.random.default_rng([SEED, config.d, 1, 2])

    cached = None if overwrite else _load_calibration(config, calibration_depth)
    if cached is not None:
        num_buffers, n_penalize = cached
        # Regenerate the same calibration targets the probe needs. These come from
        # a deterministically seeded RNG, so this reproduces them without rerunning
        # any optimization.
        calibration_targets = [
            sample_target(config, calibration_rng)
            for _ in range(N_CALIBRATION_CIRCUITS)
        ]
        print(
            "\n  [compile] using cached calibration  "
            f"{config.key} k{calibration_depth}  "
            f"num_buffers={num_buffers}  n_pen={n_penalize}"
        )
    else:
        num_buffers, n_penalize, calibration_targets = _calibrate_buffers(
            config, calibration_depth, calibration_rng, n_jobs=n_jobs,
            optimizer=optimizer,
        )
    n_cavity = config.d + num_buffers
    if not use_probe:
        # Skip the probe and start production at the calibration depth. That depth
        # is chosen to be comfortably deep, so most targets converge on the first k
        # instead of burning attempts on shallow depths that fail.
        production_start = calibration_depth
        print(
            "\n  [compile] depth probe disabled  "
            f"{config.key}  k_start={production_start} (= k_cal)"
        )
    else:
        cached_start = (
            None if overwrite
            else _load_probe(config, depth_floor, n_cavity, n_penalize, optimizer)
        )
        if cached_start is not None:
            production_start = cached_start
            print(
                "\n  [compile] using cached depth probe  "
                f"{config.key}  k_start={production_start}"
            )
        else:
            production_start = _probe_minimum_depth(
                calibration_targets,
                config,
                depth_floor,
                calibration_depth,
                n_cavity,
                n_penalize,
                probe_rng,
                n_jobs,
                optimizer=optimizer,
            )
    output_path = compiled_path(config, production_start)
    if os.path.exists(output_path) and not overwrite:
        print(f"  [compile] cached -> {os.path.basename(output_path)}")
        return output_path

    maximum_depth = calibration_depth * K_MAX_FACTOR
    sizing = f"batch={BATCH_SIZE}" if optimizer == "adam" else "restart-until-converged"
    print(
        "\n  [compile] production  "
        f"N_cav={n_cavity}  k_start={production_start}  k_max={maximum_depth}  "
        f"N_unitaries={N_UNITARIES}  optimizer={optimizer}  {sizing}  N_JOBS={n_jobs}"
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
                optimizer=optimizer,
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
            "k": production_start,
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
