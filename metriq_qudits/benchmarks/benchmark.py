from __future__ import annotations

from pydantic import BaseModel

from metriq_qudits.benchmarks.helpers import (
    N_JOBS,
    build_circuit_pulses,
    load_result,
    save_result,
    simulate_circuits,
)
from metriq_qudits.compilation.circuit_io import load_circuits, save_circuits
from metriq_qudits.paths import data_dir
from metriq_qudits.pulses.pulse_io import load_pulses, save_pulses
from metriq_qudits.system_config import SystemConfig


class BenchmarkResult(BaseModel):
    """Typed benchmark metrics. Subclasses declare their metric fields."""


class Benchmark:
    """Runs the staged, existence-cached local pipeline for one benchmark.

    Concrete benchmarks supply four hooks (sample_targets, calibrate, compile,
    score); the base owns the compile -> build -> simulate -> score orchestration,
    the artifact layout, and the T1/T2 sweep. A run pairs a config (one qudit
    dimension) with a device (backend and noise grid)."""

    result_cls: type[BenchmarkResult]

    def __init__(self, args, params: BaseModel) -> None:
        self.args = args
        self.params = params

    # Hooks implemented by concrete benchmarks.
    def sample_targets(self, config: SystemConfig):
        raise NotImplementedError

    def calibrate(self, config: SystemConfig):
        """Return the compile/simulate policy. Must expose a .n_cavity truncation."""
        raise NotImplementedError

    def compile(self, config: SystemConfig, targets, cal):
        raise NotImplementedError

    def score(self, config: SystemConfig, states, targets, cal) -> BenchmarkResult:
        raise NotImplementedError

    def run_dir(self, config: SystemConfig):
        return data_dir("runs", self.params.benchmark_name, config.key)

    def run(self, config: SystemConfig, device: BaseModel) -> BenchmarkResult:
        """Execute compilation, pulse construction, simulation, and scoring."""
        overwrite = getattr(self.args, "overwrite", False)
        n_jobs = getattr(self.args, "n_jobs", N_JOBS)
        correct_phases = getattr(self.args, "correct_phases", True)
        n_unitaries = self.params.n_unitaries

        cal = self.calibrate(config)

        circuits_dir = self.run_dir(config) / "circuits"
        pulses_dir = self.run_dir(config) / "pulses"
        circuits_dir.mkdir(parents=True, exist_ok=True)
        pulses_dir.mkdir(parents=True, exist_ok=True)
        circuit_paths = [circuits_dir / f"{i:04d}.npz" for i in range(n_unitaries)]
        pulse_paths = [pulses_dir / f"{i:04d}.npz" for i in range(n_unitaries)]

        # Compile stage: compile the ensemble, then save every circuit.
        # Overwrite gate: run when forced or missing, otherwise reuse what is on disk.
        if overwrite or not all(path.exists() for path in circuit_paths):
            targets = self.sample_targets(config)
            circuits = self.compile(config, targets, cal)
            save_circuits(circuit_paths, circuits)
        circuits = load_circuits(circuit_paths)

        # Pulse stage: build waveforms from the circuits, then save every pulse.
        compiled = [i for i, circuit in enumerate(circuits) if circuit is not None]
        if overwrite or not all(pulse_paths[i].exists() for i in compiled):
            pulses = build_circuit_pulses(circuits, correct_phases=correct_phases, n_jobs=n_jobs)
            save_pulses(pulse_paths, pulses)

        # Targets to score against, recovered from each compiled circuit.
        targets = [circuit.target_state if circuit is not None else None
                   for circuit in circuits]

        # Circuits and pulses are noise-independent, so scores are namespaced per device.
        metrics_dir = self.run_dir(config) / "metrics" / device.name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Simulation stage (noiseless): simulate each pulse, then score separately.
        noiseless_path = metrics_dir / "noiseless.npz"
        if overwrite or not noiseless_path.exists():
            pulses = load_pulses(pulse_paths)
            states = simulate_circuits(pulses, n_cavity=cal.n_cavity,
                                       backend=device.backend, n_jobs=n_jobs)
            save_result(noiseless_path, self.score(config, states, targets, cal))
        result = load_result(noiseless_path, self.result_cls)
        print(f"  [noiseless] {config.key}  HOG={result.hog_mean:.4f}  "
              f"XEB={result.xeb_normalized:.4f}  F={result.fid_mean:.4f}")

        # Noise sweep: one aggregated file per (T1, T2) grid point, keeping T2 <= 2 T1.
        # Empty grids (an ideal device) run the noiseless baseline only.
        if device.t1_us and device.t2_us:
            sweep_dir = metrics_dir / "sweep"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            pulses = load_pulses(pulse_paths)
            for i1, t1_us in enumerate(device.t1_us):
                for i2, t2_us in enumerate(device.t2_us):
                    if t2_us > 2 * t1_us:
                        continue
                    point_path = sweep_dir / f"{i1:02d}_{i2:02d}.npz"
                    if not overwrite and point_path.exists():
                        continue
                    states = simulate_circuits(pulses, n_cavity=cal.n_cavity,
                                               backend=device.backend,
                                               t1_us=t1_us, t2_us=t2_us, n_jobs=n_jobs)
                    point = self.score(config, states, targets, cal)
                    save_result(point_path, point)
                    print(f"  [sweep] T1={t1_us:.0f}us T2={t2_us:.0f}us  "
                          f"HOG={point.hog_mean:.3f}  XEB={point.xeb_normalized:.3f}  "
                          f"F={point.fid_mean:.3f}")
        return result
