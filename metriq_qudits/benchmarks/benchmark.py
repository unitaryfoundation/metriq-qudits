from __future__ import annotations

import time

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
from metriq_qudits.parallel import milestone_progress
from metriq_qudits.system_config import SystemConfig


class BenchmarkResult(BaseModel):
    """Typed benchmark metrics. Subclasses declare their metric fields."""


class Benchmark:
    """Runs the local pipeline for one benchmark."""

    result_cls: type[BenchmarkResult]

    def __init__(self, args, params: BaseModel) -> None:
        self.args = args
        self.params = params

    # Instantiated by concrete benchmarks (e.g. quantum volume).
    def sample_targets(self, config: SystemConfig):
        raise NotImplementedError

    def calibrate(self, config: SystemConfig):
        raise NotImplementedError

    def compile(self, config: SystemConfig, targets, cal, on_progress=None):
        raise NotImplementedError

    def score(self, config: SystemConfig, states, targets, cal) -> BenchmarkResult:
        raise NotImplementedError

    def run_dir(self, config: SystemConfig):
        return data_dir("runs", self.params.benchmark_name, config.key)

    @staticmethod
    def _detail(message: str) -> None:
        print(f"      {message}", flush=True)

    @staticmethod
    def _elapsed(start: float) -> str:
        return f"{time.perf_counter() - start:.1f}s"

    def run(self, config: SystemConfig, device: BaseModel) -> BenchmarkResult:
        """Execute compilation, pulse construction, simulation, and scoring."""
        overwrite = getattr(self.args, "overwrite", False)
        n_jobs = getattr(self.args, "n_jobs", N_JOBS)
        correct_phases = getattr(self.args, "correct_phases", True)
        n_unitaries = self.params.n_unitaries

        print("  [1/4] Calibration", flush=True)
        calibration_path = self.run_dir(config) / "calibration.npz"
        calibration_cached = not overwrite and calibration_path.exists()
        calibration_start = time.perf_counter()
        if not calibration_cached:
            self._detail(f"selecting cavity truncation and layer window ({n_jobs} worker(s))")
        cal = self.calibrate(config)
        if calibration_cached:
            self._detail(
                f"cache hit · cavity N={cal.n_cavity}, penalized levels={cal.n_penalize}, "
                f"layer window k={cal.k_start}..{cal.k_max}"
            )
        else:
            self._detail(
                f"complete in {self._elapsed(calibration_start)} · cavity N={cal.n_cavity}, "
                f"penalized levels={cal.n_penalize}, layer window k={cal.k_start}..{cal.k_max}"
            )

        circuits_dir = self.run_dir(config) / "circuits"
        pulses_dir = self.run_dir(config) / "pulses"
        circuits_dir.mkdir(parents=True, exist_ok=True)
        pulses_dir.mkdir(parents=True, exist_ok=True)
        circuit_paths = [circuits_dir / f"{i:04d}.npz" for i in range(n_unitaries)]
        pulse_paths = [pulses_dir / f"{i:04d}.npz" for i in range(n_unitaries)]

        # Compile stage: compile the ensemble, then save every circuit.
        # Overwrite gate: run when forced (i.e. --overwrite True) or missing, otherwise reuse what is on disk.
        print("  [2/4] Circuit compilation", flush=True)
        circuits_cached = not overwrite and all(path.exists() for path in circuit_paths)
        compile_start = time.perf_counter()
        if not circuits_cached:
            self._detail(
                f"optimizing {n_unitaries} Haar-random circuits ({n_jobs} worker(s))"
            )
            targets = self.sample_targets(config)
            circuits = self.compile(
                config, targets, cal,
                on_progress=milestone_progress("compilation jobs"),
            )
            save_circuits(circuit_paths, circuits)
        circuits = load_circuits(circuit_paths)
        n_compiled = sum(circuit is not None for circuit in circuits)
        if circuits_cached:
            self._detail(f"cache hit · {n_compiled}/{n_unitaries} circuits available")
        else:
            self._detail(
                f"complete in {self._elapsed(compile_start)} · "
                f"{n_compiled} succeeded, {n_unitaries - n_compiled} failed"
            )

        # Pulse stage: build waveforms from the circuits, then save every pulse.
        compiled = [i for i, circuit in enumerate(circuits) if circuit is not None]
        print("  [3/4] Pulse construction", flush=True)
        pulses_cached = not overwrite and all(pulse_paths[i].exists() for i in compiled)
        pulse_start = time.perf_counter()
        if not pulses_cached:
            self._detail(f"building {len(compiled)} pulse sequences ({n_jobs} worker(s))")
            pulses = build_circuit_pulses(circuits, correct_phases=correct_phases, n_jobs=n_jobs)
            save_pulses(pulse_paths, pulses)
        n_pulses = sum(pulse_paths[i].exists() for i in compiled)
        if pulses_cached:
            self._detail(f"cache hit · {n_pulses}/{len(compiled)} pulses available")
        else:
            self._detail(
                f"complete in {self._elapsed(pulse_start)} · "
                f"{n_pulses}/{len(compiled)} pulses built"
            )

        # Targets to score against, recovered from each compiled circuit.
        targets = [circuit.target_state if circuit is not None else None
                   for circuit in circuits]

        # Circuits and pulses are noise-independent, so scores are namespaced per device.
        metrics_dir = self.run_dir(config) / "metrics" / device.name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Simulation stage (noiseless): simulate each pulse, then score separately.
        print("  [4/4] Simulation and scoring", flush=True)
        noiseless_path = metrics_dir / "noiseless.npz"
        noiseless_cached = not overwrite and noiseless_path.exists()
        noiseless_start = time.perf_counter()
        if not noiseless_cached:
            self._detail(
                f"noiseless simulation of {n_pulses} pulses on {device.backend} "
                f"({n_jobs} worker(s))"
            )
            pulses = load_pulses(pulse_paths)
            states = simulate_circuits(pulses, n_cavity=cal.n_cavity,
                                       backend=device.backend, n_jobs=n_jobs,
                                       on_progress=milestone_progress("simulation jobs"))
            save_result(noiseless_path, self.score(config, states, targets, cal))
            self._detail(f"noiseless simulation complete in {self._elapsed(noiseless_start)}")
        else:
            self._detail("noiseless simulation cache hit")
        result = load_result(noiseless_path, self.result_cls)
        self._detail(
            f"noiseless metrics · HOG={result.hog_mean:.4f}, "
            f"XEB={result.xeb_normalized:.4f}, fidelity={result.fid_mean:.4f}"
        )

        # Noise sweep: one aggregated file per (T1, T2) grid point, keeping T2 <= 2 T1.
        # When (T1, T2) grid is empty (an ideal device), run the noiseless baseline only.
        if device.t1_us and device.t2_us:
            sweep_dir = metrics_dir / "sweep"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            pulses = load_pulses(pulse_paths)
            sweep_points = [
                (i1, i2, t1_us, t2_us)
                for i1, t1_us in enumerate(device.t1_us)
                for i2, t2_us in enumerate(device.t2_us)
                if t2_us <= 2 * t1_us
            ]
            point_paths = [sweep_dir / f"{i1:02d}_{i2:02d}.npz"
                           for i1, i2, _, _ in sweep_points]
            n_cached = (0 if overwrite else
                        sum(path.exists() for path in point_paths))
            n_to_run = len(sweep_points) - n_cached
            self._detail(
                f"noise sweep · {len(sweep_points)} valid points, "
                f"{n_cached} cached, {n_to_run} to simulate"
            )
            sweep_start = time.perf_counter()
            n_finished = n_cached
            for (i1, i2, t1_us, t2_us), point_path in zip(sweep_points, point_paths):
                if not overwrite and point_path.exists():
                    continue
                point_start = time.perf_counter()
                states = simulate_circuits(pulses, n_cavity=cal.n_cavity,
                                           backend=device.backend,
                                           t1_us=t1_us, t2_us=t2_us, n_jobs=n_jobs)
                point = self.score(config, states, targets, cal)
                save_result(point_path, point, t1_us=t1_us, t2_us=t2_us)
                n_finished += 1
                self._detail(
                    f"sweep {n_finished}/{len(sweep_points)} · "
                    f"T1={t1_us:.0f}us, T2={t2_us:.0f}us · "
                    f"HOG={point.hog_mean:.3f}, XEB={point.xeb_normalized:.3f}, "
                    f"fidelity={point.fid_mean:.3f} · {self._elapsed(point_start)}"
                )
            if n_to_run:
                self._detail(
                    f"noise sweep complete in {self._elapsed(sweep_start)} · "
                    f"{n_to_run} simulated, {n_cached} reused"
                )
            else:
                self._detail("noise sweep cache hit · all points reused")
        else:
            self._detail("noise sweep not requested")
        return result
