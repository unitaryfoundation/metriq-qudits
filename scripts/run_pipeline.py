"""Run the ECD quantum-volume experiment from compilation through simulation.

Example uses
``python scripts/run_pipeline.py --configs d4_m1``
``N_JOBS=8 python scripts/run_pipeline.py --skip-sweep``
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecd_qv.compilation.compile import compile_circuits
from ecd_qv.pulses.pulse_stage import (
    CHI_KHZ,
    CHI_PRIME_HZ,
    SELF_KERR_HZ,
    build_circuit_pulses,
)
from ecd_qv.system_config import SystemConfig
from ecd_qv.simulation.sweep import T1_VALUES, T2_VALUES, run_noise_sweep, run_noiseless

N_JOBS = int(os.environ.get("N_JOBS", "1"))

CONFIGS = (
    *(SystemConfig(d=d) for d in (4, 6, 8, 10, 12, 14, 16)), # User can modify dimensions to sweep over
)
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
    """Execute the four stages for one logical system."""
    # Stage 1: gate-level ECD decomposition
    compiled_path = compile_circuits(
        config, overwrite=overwrite, n_jobs=n_jobs,
    )

    # Stage 2: Optimize the pulses making up the ECD gates to obtain the
    #          desired beta displacements for the given system config.
    pulse_path = build_circuit_pulses(
        compiled_path,
        correct_phases=correct_phases,
        overwrite=overwrite,
        n_jobs=n_jobs,
    )

    # Stage 3: Run noiseless and (optionally) noisy simulations
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

        # Run all stages of pipeline
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
        from plot_results import main as plot_main

        plot_main()


if __name__ == "__main__":
    main()
