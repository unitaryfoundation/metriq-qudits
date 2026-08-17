from __future__ import annotations

import os
import platform

# JAX can otherwise select an unsupported backend while importing the compiler.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass

import numpy as np
from scipy.stats import unitary_group

from metriq_qudits.benchmarks.benchmark import Benchmark, BenchmarkResult
from metriq_qudits.benchmarks.helpers import N_JOBS, compile_circuits
from metriq_qudits.compilation.ecd_parameter_finder import OptimizerConfig
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


@dataclass(frozen=True)
class Calibration:
    """Per-config Fock truncation, penalty count, and depth window."""

    n_cavity: int
    n_penalize: int
    k_start: int
    k_max: int


class QuantumVolume(Benchmark):
    result_cls = QVResult

    def sample_targets(self, config: SystemConfig) -> list[np.ndarray]:
        """Draw the ensemble of targets |psi> = U|0> from the Haar measure."""
        rng = np.random.default_rng([self.params.seed, config.d])
        return [unitary_group.rvs(config.dimension, random_state=rng)[:, 0]
                for _ in range(self.params.n_unitaries)]

    def calibrate(self, config: SystemConfig) -> Calibration:
        # Provisional truncation and depth window (calibration pass sets these later).
        d = config.d
        return Calibration(n_cavity=d + 4, n_penalize=0, k_start=1, k_max=4 * d)

    def compile(self, config: SystemConfig, targets, cal: Calibration):
        return compile_circuits(
            targets, d=config.d, n_cavity=cal.n_cavity, k_init=cal.k_start,
            k_max=cal.k_max, optimizer_config=OptimizerConfig(n_penalize=cal.n_penalize),
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
