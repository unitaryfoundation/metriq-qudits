"""Run noiseless and T1/T2 simulations from cached circuit waveforms."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ecd_qv.benchmark.circuit_io import load_circuits
from ecd_qv.benchmark.metrics import eval_circuit
from ecd_qv.parallel import parallel_map_unordered
from ecd_qv.pulses.pulse_stage import (
    CHI_PRIME_RAD_S,
    CHI_RAD_S,
    SELF_KERR_RAD_S,
    make_storages,
    physics_metadata,
    physics_metadata_matches,
    variant_suffix,
)
from ecd_qv.pulses.pulse_io import load_pulses
from ecd_qv.system_config import SystemConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = REPO_ROOT / ".cache" / "noise_sweep_pulse"
NOISELESS_DIR = REPO_ROOT / ".cache" / "noiseless"

T1_VALUES = np.array([5, 10, 20, 50, 100]) * 1e-6
T2_VALUES = np.array([10, 20, 40, 100, 200]) * 1e-6
N_CAVITY_SIMULATION = None  # None uses the compiler's calibrated truncation.
METRICS = ("hog", "xeb", "fid")


def results_path(
    directory: Path,
    prefix: str,
    config: SystemConfig,
    correct_phases: bool,
) -> str:
    suffix = variant_suffix(correct_phases)
    return str(directory / f"{prefix}_{config.key}{suffix}.npz")


def _make_simulator(backend, n_cavity):
    if backend == "dynamiqs":
        from ecd_qv.simulation.displaced_frame_simulator_dq import DisplacedFrameSimulatorDQ

        simulator_class = DisplacedFrameSimulatorDQ
    else:
        from ecd_qv.simulation.displaced_frame_simulator import DisplacedFrameSimulator

        simulator_class = DisplacedFrameSimulator
    return simulator_class(
        cavity_dim=n_cavity,
        storage=make_storages(1)[0],
    )


def _simulate_noise_point(args):
    """Simulate every circuit at one ``(T1, T2)`` point."""
    (
        pulses,
        target_states,
        n_cavity,
        d,
        t1,
        t2,
        backend,
    ) = args
    simulator = _make_simulator(backend, n_cavity)
    t1_us = t1 * 1e6 if t1 is not None else None
    t2_us = t2 * 1e6 if t2 is not None else None

    values = {metric: [] for metric in METRICS}
    for pulse, target_state in zip(pulses, target_states):
        result, alpha = simulator.simulate(
            epsilon=pulse.cavity_drives[0],
            omega=pulse.ancilla_drive,
            T1_us=t1_us,
            T2_us=t2_us,
        )
        physical_state = simulator.to_physical_frame(
            result.states[-1],
            alpha,
            cavity_phase=pulse.final_cavity_phases[0],
        )
        cavity_state = physical_state.ptrace([0]).full()
        metrics = eval_circuit(
            cavity_state,
            target_state,
            d,
            1,
            n_cavity,
        )
        for metric, value in zip(METRICS, metrics):
            values[metric].append(value)
    return values


def _load_inputs(pulse_path: str, compiled_path: str):
    circuits, metadata = load_circuits(compiled_path)
    pulses, _ = load_pulses(pulse_path)
    if int(metadata["num_modes"]) != 1:
        raise ValueError("expected a single-qudit cache")
    config = SystemConfig(d=int(metadata["d"]))
    if len(pulses) != len(circuits):
        raise ValueError("pulse and compiled-circuit counts differ")
    if any(pulse.num_modes != 1 for pulse in pulses):
        raise ValueError("pulse cache has the wrong number of cavity modes")
    n_cavity = (
        N_CAVITY_SIMULATION
        if N_CAVITY_SIMULATION is not None
        else int(metadata["N_cav"])
    )
    return config, n_cavity, circuits, pulses, metadata


def run_noiseless(
    pulse_path: str,
    compiled_path: str,
    *,
    correct_phases: bool = True,
    backend: str = "dynamiqs",
    diagnostics_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
) -> str:
    """Run the three-level, closed-system reference simulation."""
    if diagnostics_dir is not None:
        from ecd_qv.plot_utils import save_state_diagnostics

    config, n_cavity, circuits, pulses, _ = _load_inputs(
        pulse_path, compiled_path,
    )
    NOISELESS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = results_path(
        NOISELESS_DIR,
        "noiseless",
        config,
        correct_phases,
    )
    if os.path.exists(output_path) and not overwrite:
        with np.load(output_path) as cached:
            cache_matches = physics_metadata_matches(
                cached, correct_phases,
            )
            if cache_matches:
                print(
                    f"  [noiseless] cached  d={config.d}  "
                    f"HOG={float(cached['hog_mean']):.4f}  "
                    f"XEB={float(cached['xeb_mean']):.4f}  "
                    f"F={float(cached['fid_mean']):.4f}  N_cav={n_cavity}"
                )
        if cache_matches:
            return output_path
        print("  [noiseless] cached physics changed - rerunning")

    simulator = _make_simulator(backend, n_cavity)
    print(
        f"\n  [noiseless] d={config.d}  "
        f"N_cav={n_cavity}  ({len(circuits)} circuits)  backend={backend}"
    )
    print(
        f"  {'ci':>3}  {'k':>3}  {'HOG':>8}  {'XEB':>8}  {'F':>8}  "
        f"{'P(g)':>6}  {'peak_n':>7}  {'max|beta|':>9}"
    )

    values = {metric: [] for metric in METRICS}
    start = time.perf_counter()
    for index, (pulse, circuit) in enumerate(zip(pulses, circuits)):
        result, alpha = simulator.simulate(
            epsilon=pulse.cavity_drives[0],
            omega=pulse.ancilla_drive,
            T1_us=None,
            T2_us=None,
        )
        final_state = simulator.to_physical_frame(
            result.states[-1],
            alpha,
            cavity_phase=pulse.final_cavity_phases[0],
        )
        if diagnostics_dir is not None:
            save_state_diagnostics(
                alpha,
                final_state,
                diagnostics_dir,
                psi_target=circuit.target_state,
                N_cav=n_cavity,
                circuit_idx=index,
            )

        cavity_state = final_state.ptrace([0]).full()
        metrics = eval_circuit(
            cavity_state,
            circuit.target_state,
            config.d,
            1,
            n_cavity,
        )
        for metric, value in zip(METRICS, metrics):
            values[metric].append(value)

        p_ground = float(final_state.ptrace([1])[0, 0].real)
        peak_photons = float(np.max(np.abs(alpha) ** 2))
        print(
            f"    {index + 1:>3}  {circuit.depth:>3}  "
            f"{metrics[0]:>8.4f}  {metrics[1]:>8.4f}  {metrics[2]:>8.4f}  "
            f"{p_ground:>6.3f}  {peak_photons:>7.1f}  "
            f"{float(np.max(np.abs(circuit.betas))):>9.3f}",
            flush=True,
        )

    time_per_circuit = (time.perf_counter() - start) / len(circuits)
    n_sweep_points = sum(
        t2 <= 2 * t1 for t1 in T1_VALUES for t2 in T2_VALUES
    )
    estimated_hours = (
        n_sweep_points * len(circuits) * time_per_circuit / max(n_jobs, 1) / 3600
    )
    print(
        f"    {time_per_circuit:.1f}s/circuit -> projected full T1/T2 sweep "
        f"~ {estimated_hours:.1f} h at N_JOBS={n_jobs}"
    )

    np.savez(
        output_path,
        **{
            f"{metric}_mean": float(np.mean(values[metric]))
            for metric in METRICS
        },
        **{
            f"{metric}_per_circuit": np.asarray(values[metric])
            for metric in METRICS
        },
        d=config.d,
        num_modes=1,
        ansatz="haar",
        N_cav=n_cavity,
        N_unitaries=len(circuits),
        **physics_metadata(correct_phases),
    )
    summary = "  ".join(
        f"{metric.upper()}={np.mean(values[metric]):.4f}" for metric in METRICS
    )
    print(f"  [noiseless] d={config.d}  {summary}  -> {os.path.basename(output_path)}")
    return output_path


def run_noise_sweep(
    pulse_path: str,
    compiled_path: str,
    *,
    correct_phases: bool = True,
    backend: str = "dynamiqs",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> str:
    """Run or resume the three-level ancilla T1/T2 sweep."""
    config, n_cavity, circuits, pulses, metadata = _load_inputs(
        pulse_path, compiled_path,
    )
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = results_path(
        SWEEP_DIR,
        "results",
        config,
        correct_phases,
    )
    print(
        f"\n  [sweep] d={config.d}  "
        f"k={int(metadata['k'])}  N_cav={n_cavity}  ({len(circuits)} circuits)"
    )
    print(
        f"    chi/2pi={CHI_RAD_S / (2 * np.pi * 1e3):.1f} kHz  "
        f"K/2pi={SELF_KERR_RAD_S / (2 * np.pi):.1f} Hz  "
        f"chi'/2pi={CHI_PRIME_RAD_S / (2 * np.pi):.1f} Hz"
    )

    shape = (len(T1_VALUES), len(T2_VALUES))
    means = {metric: np.full(shape, np.nan) for metric in METRICS}
    standard_deviations = {
        metric: np.full(shape, np.nan) for metric in METRICS
    }
    if os.path.exists(output_path) and not overwrite:
        with np.load(output_path) as previous:
            axes_match = (
                np.array_equal(previous["T1_values"], T1_VALUES)
                and np.array_equal(previous["T2_values"], T2_VALUES)
            )
            cache_matches = physics_metadata_matches(
                previous, correct_phases,
            )
            if axes_match and cache_matches:
                for metric in METRICS:
                    means[metric] = previous[metric][:, :, 0]
                    standard_deviations[metric] = previous[f"{metric}_std"][:, :, 0]
        if axes_match and cache_matches:
            complete = int(np.sum(~np.isnan(means["hog"])))
            print(f"    resuming: {complete} grid point(s) already complete")
        elif not cache_matches:
            print("    cached physics changed - starting fresh")
        else:
            print("    T1/T2 axes changed - starting fresh")

    grid_points = [
        (i1, i2, t1, t2)
        for i1, t1 in enumerate(T1_VALUES)
        for i2, t2 in enumerate(T2_VALUES)
        if t2 <= 2 * t1 and np.isnan(means["hog"][i1, i2])
    ]
    if not grid_points:
        print("    all grid points complete - skipping")
        return output_path
    print(f"    {len(grid_points)} grid point(s) remaining  N_JOBS={n_jobs}")

    def save_checkpoint(i1, i2, values, elapsed):
        ddof = 1 if len(values["hog"]) > 1 else 0
        for metric in METRICS:
            means[metric][i1, i2] = np.mean(values[metric])
            standard_deviations[metric][i1, i2] = np.std(
                values[metric], ddof=ddof,
            )
        summary = "  ".join(
            f"{metric.upper()}={means[metric][i1, i2]:.3f}"
            f"+/-{standard_deviations[metric][i1, i2]:.3f}"
            for metric in METRICS
        )
        print(
            f"    T1={T1_VALUES[i1] * 1e6:.0f}us  "
            f"T2={T2_VALUES[i2] * 1e6:.0f}us  {summary}  ({elapsed:.1f}s)",
            flush=True,
        )
        np.savez(
            output_path,
            T1_values=T1_VALUES,
            T2_values=T2_VALUES,
            **{
                metric: means[metric][:, :, np.newaxis]
                for metric in METRICS
            },
            **{
                f"{metric}_std": standard_deviations[metric][:, :, np.newaxis]
                for metric in METRICS
            },
            d=config.d,
            num_modes=1,
            k=int(metadata["k"]),
            ansatz="haar",
            N_cav=n_cavity,
            N_unitaries=len(circuits),
            **physics_metadata(correct_phases),
        )

    target_states = [circuit.target_state for circuit in circuits]
    jobs = {
        (i1, i2): (
            pulses,
            target_states,
            n_cavity,
            config.d,
            t1,
            t2,
            backend,
        )
        for i1, i2, t1, t2 in grid_points
    }
    start = time.perf_counter()
    for (i1, i2), values in parallel_map_unordered(
        _simulate_noise_point, jobs, n_jobs,
    ):
        save_checkpoint(i1, i2, values, time.perf_counter() - start)

    print(f"  Saved -> {output_path}")
    return output_path
