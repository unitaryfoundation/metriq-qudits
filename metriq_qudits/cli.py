"""Command-line interface for the metriq-qudits benchmarks.

Examples:
    # Run a benchmark config against the default ideal (noiseless) device
    metriq-qudits schemas/examples/quantum_volume.example.json

    # Run against a simulated device with a T1/T2 sweep, in parallel
    metriq-qudits my_qv.json --device schemas/examples/coherence_sweep.device.json --n-jobs 8

    # Force a fresh run, ignoring cached stage outputs
    metriq-qudits my_qv.json --overwrite --output-dir /tmp/run
"""

from __future__ import annotations

import argparse
import os
import platform
import time

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from metriq_qudits.constants import JobType
from metriq_qudits.paths import output_dir, set_output_dir
from metriq_qudits.registry import BENCHMARK_HANDLERS
from metriq_qudits.schema_validator import load_and_validate, load_device, validate_device
from metriq_qudits.system_config import SystemConfig


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a metriq-qudits benchmark from a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        metavar="BENCHMARK_CONFIG",
        help="Benchmark config JSON (see metriq_qudits/schemas/examples).",
    )
    parser.add_argument(
        "--device",
        metavar="PATH",
        help="Simulated-device config JSON. Defaults to an ideal (noiseless) device.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("N_JOBS", "1")),
        metavar="N",
        help="Parallel worker processes. Defaults to the N_JOBS environment variable (or 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore cached artifacts and rerun every stage.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help="Artifact root. Defaults to METRIQ_QUDITS_OUTPUT_DIR or ./outputs.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.output_dir is not None:
        set_output_dir(args.output_dir)

    params = load_and_validate(args.config)
    device = load_device(args.device) if args.device else validate_device({"name": "ideal"})
    benchmark = BENCHMARK_HANDLERS[JobType(params.benchmark_name)](args=args, params=params)

    print(
        f"benchmark={params.benchmark_name}  device={device.name}  backend={device.backend}  "
        f"dimensions={params.dimensions}  n_unitaries={params.n_unitaries}  n_jobs={args.n_jobs}"
    )
    print(f"Output directory: {output_dir()}")

    start = time.perf_counter()
    for d in params.dimensions:
        print("\n" + "=" * 60)
        print(f"  d={d}")
        print("=" * 60)
        benchmark.run(SystemConfig(d=d), device)
    print("\n" + "=" * 60)
    print(f"  Done in {time.perf_counter() - start:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
