"""Command-line interface for the ECD quantum-volume pipeline."""

from __future__ import annotations

import argparse
import os
import platform
import time

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from metriq_qudits.compilation.compile import compile_circuits
from metriq_qudits.paths import output_dir, set_output_dir
from metriq_qudits.pulses.pulse_stage import (
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
) -> None:
    """Execute compilation, pulse construction, and simulation."""
    compiled_path = compile_circuits(config, overwrite=overwrite, n_jobs=n_jobs)

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
    )
    if include_noise_sweep:
        run_noise_sweep(
            pulse_path,
            compiled_path,
            correct_phases=correct_phases,
            backend=backend,
            overwrite=overwrite,
            n_jobs=n_jobs,
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
        "--plot",
        action="store_true",
        help="Regenerate result figures from the cache after the run.",
    )
    return parser, parser.parse_args(argv)


def main(argv=None) -> None:
    parser, args = _parse_args(argv)
    if args.output_dir is not None:
        set_output_dir(args.output_dir)
    if args.configs:
        unknown = [key for key in args.configs if key not in CONFIG_BY_KEY]
        if unknown:
            parser.error(
                f"Unknown config key(s): {unknown}. Available: {list(CONFIG_BY_KEY)}"
            )
        selected = [CONFIG_BY_KEY[key] for key in args.configs]
    else:
        selected = CONFIGS

    correct_phases = not args.no_phase_correction
    print(
        f"N_JOBS={N_JOBS}  configs={[config.key for config in selected]}  "
        f"backend={args.backend}"
    )
    print(f"Output directory: {output_dir()}")
    print(
        f"Physics: chi/2pi={CHI_KHZ:.1f} kHz  "
        f"K/2pi={SELF_KERR_HZ:.1f} Hz  chi'/2pi={CHI_PRIME_HZ:.1f} Hz"
    )
    print(f"T1 grid: {T1_VALUES * 1e6} us")
    print(f"T2 grid: {T2_VALUES * 1e6} us")

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
            n_jobs=N_JOBS,
        )
    print("\n" + "=" * 60)
    print(f"  Done in {time.perf_counter() - start:.1f}s")
    print("=" * 60)

    if args.plot:
        print("\nGenerating result figures ...")
        from metriq_qudits.plot_results import main as plot_main

        plot_main()


if __name__ == "__main__":
    main()
