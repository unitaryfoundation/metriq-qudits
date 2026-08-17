from __future__ import annotations

import os
import platform

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.stats import unitary_group

from metriq_qudits.benchmarks.benchmark import Benchmark, BenchmarkResult
from metriq_qudits.benchmarks.helpers import N_JOBS, compile_circuits
from metriq_qudits.compilation.calibration import (
    Calibration,
    compile_config,
    load_or_calibrate,
)
from metriq_qudits.metrics import eval_circuit
from metriq_qudits.system_config import SystemConfig


class QVResult(BenchmarkResult):
    """Per-circuit HOG, XEB, and fidelity for one simulated ensemble."""

    hog: list[float]
    xeb: list[float]
    fid: list[float]

    @property
    def hog_mean(self) -> float:
        return float(np.mean(self.hog))

    @property
    def xeb_mean(self) -> float:
        return float(np.mean(self.xeb))

    @property
    def fid_mean(self) -> float:
        return float(np.mean(self.fid))


class QuantumVolume(Benchmark):
    result_cls = QVResult

    def sample_targets(self, config: SystemConfig) -> list[np.ndarray]:
        """Draw the ensemble of targets |psi> = U|0> from the Haar measure."""
        rng = np.random.default_rng([self.params.seed, config.d])
        return [unitary_group.rvs(config.dimension, random_state=rng)[:, 0]
                for _ in range(self.params.n_unitaries)]

    def calibrate(self, config: SystemConfig) -> Calibration:
        return load_or_calibrate(
            self.run_dir(config) / "calibration.npz", config.d,
            overwrite=getattr(self.args, "overwrite", False),
            n_jobs=getattr(self.args, "n_jobs", N_JOBS),
        )

    def compile(self, config: SystemConfig, targets, cal: Calibration):
        return compile_circuits(
            targets, d=config.d, n_cavity=cal.n_cavity, k_init=cal.k_start,
            k_max=cal.k_max, optimizer_config=compile_config(cal),
            n_jobs=getattr(self.args, "n_jobs", N_JOBS),
        )

    def score(self, config: SystemConfig, states, targets, cal: Calibration) -> QVResult:
        metrics = [eval_circuit(state, psi, config.d, 1, cal.n_cavity)
                   for state, psi in zip(states, targets)
                   if state is not None and psi is not None]
        if not metrics:
            raise ValueError("no circuits were scored")
        hog, xeb, fid = zip(*metrics)
        return QVResult(hog=list(hog), xeb=list(xeb), fid=list(fid))
