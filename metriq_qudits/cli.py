"""Command-line interface for the ECD quantum-volume pipeline.

Examples:
    # Smallest system, skip the T1/T2 sweep
    metriq-qudits --configs d4 --skip-sweep

    # Run several systems on the default grid

    metriq-qudits --configs d4 d6 d8

    # Custom T1/T2 sweep grids, in microseconds
    metriq-qudits --configs d4 --t1 5 10 20 50 --t2 10 20 40 100

    # Run every configured system in parallel and regenerate figures
    N_JOBS=8 metriq-qudits --plot
    metriq-qudits --plot --n-jobs 8   # same, via the CLI flag

    # Force a fresh run, ignoring cached stage outputs
    metriq-qudits --configs d4 --overwrite

    # Use the QuTiP reference backend and a custom output directory
    metriq-qudits --configs d4 --backend qutip --output-dir /tmp/run
"""

from __future__ import annotations

import argparse
import os
import platform
import time

import numpy as np

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from metriq_qudits.compilation.compile import compile_circuits
from metriq_qudits.paths import output_dir, set_output_dir
from metriq_qudits.pulses.build import (
    CHI_KHZ,
    CHI_PRIME_HZ,
    SELF_KERR_HZ,
    build_circuit_pulses,
)
from metriq_qudits.simulation.sweep import T1_VALUES, T2_VALUES, run_noise_sweep, run_noiseless
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


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ECD-QV pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help=(
            "Directory for generated artifacts. Defaults to "
            "METRIQ_QUDITS_OUTPUT_DIR or ./outputs."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        metavar="KEY",
        help=f"Configurations to run. Available: {list(CONFIG_BY_KEY)}. Default: all.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Stop after the noiseless simulation.",
    )
    parser.add_argument(
        "--t1",
        nargs="+",
        type=float,
        metavar="US",
        help=(
            "T1 grid in microseconds. Default: "
            f"{np.round(T1_VALUES * 1e6).astype(int).tolist()}."
        ),
    )
    parser.add_argument(
        "--t2",
        nargs="+",
        type=float,
        metavar="US",
        help=(
            "T2 grid in microseconds. Default: "
            f"{np.round(T2_VALUES * 1e6).astype(int).tolist()}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore cached results and rerun every stage.",
    )
    parser.add_argument(
        "--no-phase-correction",
        action="store_true",
        help="Disable the self-Kerr and second-order dispersive cavity phase corrections.",
    )
    parser.add_argument(
        "--backend",
        choices=["qutip", "dynamiqs"],
        default="dynamiqs",
        help="Physical simulator backend.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        metavar="N",
        help=(
            "Number of parallel worker processes for compilation and simulation. "
            "Defaults to the N_JOBS environment variable (or 1)."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs"],
        default="lbfgs",
        help=(
            "Parameter optimizer for compilation. 'lbfgs' (default) uses scipy "
            "L-BFGS-B, faster on CPU; 'adam' is the batched multistart optimizer."
        ),
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Run the depth probe to find a shallower production start depth. "
            "Off by default: production starts at the calibration depth (k_cal), "
            "which compiles faster but gives deeper circuits."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate result figures from the cache after the run.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate result figures from the cache and exit without running.",
    )
    return parser, parser.parse_args(argv)


def main(argv=None) -> None:
    parser, args = _parse_args(argv)
    if args.output_dir is not None:
        set_output_dir(args.output_dir)
    if args.plot_only:
        print(f"Output directory: {output_dir()}")
        print("Regenerating result figures from cache ...")
        from metriq_qudits.plotting.results import main as plot_main

        plot_main()
        return
    if args.configs:
        unknown = [key for key in args.configs if key not in CONFIG_BY_KEY]
        if unknown:
            parser.error(
                f"Unknown config key(s): {unknown}. Available: {list(CONFIG_BY_KEY)}"
            )
        selected = [CONFIG_BY_KEY[key] for key in args.configs]
    else:
        selected = CONFIGS

    t1_values = np.asarray(args.t1) * 1e-6 if args.t1 else T1_VALUES
    t2_values = np.asarray(args.t2) * 1e-6 if args.t2 else T2_VALUES
    if np.any(t1_values <= 0) or np.any(t2_values <= 0):
        parser.error("T1/T2 values must be positive.")

    correct_phases = not args.no_phase_correction
    print(
        f"N_JOBS={args.n_jobs}  configs={[config.key for config in selected]}  "
        f"backend={args.backend}  optimizer={args.optimizer}"
    )
    print(f"Output directory: {output_dir()}")
    print(
        f"Physics: chi/2pi={CHI_KHZ:.1f} kHz  "
        f"K/2pi={SELF_KERR_HZ:.1f} Hz  chi'/2pi={CHI_PRIME_HZ:.1f} Hz"
    )
    print(f"T1 grid: {t1_values * 1e6} us")
    print(f"T2 grid: {t2_values * 1e6} us")

    print("Using Gaussian ancilla pulses.")
    start = time.perf_counter()
    for config in selected:
        print("\n" + "=" * 60)
        print(
            f"  {config.key}"
            f"{'  (no phase correction)' if not correct_phases else ''}"
        )
        print("=" * 60)

        run_experiment(
            config,
            correct_phases=correct_phases,
            backend=args.backend,
            include_noise_sweep=not args.skip_sweep,
            overwrite=args.overwrite,
            n_jobs=args.n_jobs,
            optimizer=args.optimizer,
            use_probe=args.probe,
            t1_values=t1_values,
            t2_values=t2_values,
        )
    print("\n" + "=" * 60)
    print(f"  Done in {time.perf_counter() - start:.1f}s")
    print("=" * 60)

    if args.plot:
        print("\nGenerating result figures ...")
        from metriq_qudits.plotting.results import main as plot_main

        plot_main()


if __name__ == "__main__":
    main()
