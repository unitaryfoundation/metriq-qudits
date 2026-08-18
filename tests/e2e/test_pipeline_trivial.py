"""End-to-end pipeline test with a trivial circuit (known answer).

Hand-builds one compiled circuit (substituting the slow JAX optimizer), then runs
the REAL build -> simulate stages. The betas are below the ECD threshold, so the
cavity stays in vacuum and fidelity to |0> ~ 1.
"""

import numpy as np
import pytest

from metriq_qudits.benchmarks.helpers import build_circuit_pulse, simulate_circuit
from metriq_qudits.compilation.circuit_io import CompiledCircuit
from metriq_qudits.metrics import eval_circuit


def _trivial_circuit(d, depth=4):
    rng = np.random.default_rng(0)
    betas = np.full((depth, 1), 1e-6 + 0j)  # below ECD threshold -> cavity untouched
    rotations = np.column_stack([rng.uniform(0, np.pi, depth + 1),
                                 rng.uniform(0, 2 * np.pi, depth + 1)])
    target = np.zeros(d, dtype=complex)
    target[0] = 1.0  # prepare Fock |0>
    return CompiledCircuit(
        betas=betas, rotations=rotations, target_state=target,
        infidelity=0.0, boundary_leakage=0.0,
    )


def test_noiseless_pipeline_prepares_vacuum():
    d, depth, n_cav = 4, 4, 16
    circuit = _trivial_circuit(d, depth)
    pulse = build_circuit_pulse(circuit)
    state = simulate_circuit(pulse, n_cavity=n_cav, backend="qutip")
    hog, xeb, fid = eval_circuit(state, circuit.target_state, d, 1, n_cav)
    assert fid == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= hog <= 1.0
    assert xeb <= 1.0 + 1e-9
