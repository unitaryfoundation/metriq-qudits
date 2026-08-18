"""End-to-end correctness test for a single ECD gate (dynamiqs backend).

A single ECD from |g>|0> followed by a pi-pulse deterministically prepares a
coherent state with the ancilla disentangled back to |g>. The ideal cavity state
is computed independently by ``run_circuit`` (ideal ECD+R unitaries, no pulses);
the real build -> simulate (dynamiqs) pipeline must reproduce it to high fidelity.
"""

import numpy as np
import pytest

from metriq_qudits.benchmarks.helpers import build_circuit_pulse, simulate_circuit
from metriq_qudits.compilation.circuit_io import CompiledCircuit
from metriq_qudits.compilation.ecd_parameter_finder import run_circuit
from metriq_qudits.metrics import eval_circuit


def test_dq_pipeline_reproduces_single_ecd_state():
    d, n_cav = 4, 16
    beta = 0.6 + 0.75j  # |beta| ~ 0.96, the median of the compiled ensemble

    betas = np.array([[beta]])                         # depth-1 ECD
    rotations = np.array([[0.0, 0.0], [np.pi, 0.0]])   # R0 = I, then pi-flip -> ancilla |g>

    # Independent ground truth: ideal ECD+R state vector (no pulses, no noise).
    psi_g, _, _ = run_circuit(betas, rotations, n_cav)
    psi_g = np.asarray(psi_g)
    assert np.linalg.norm(psi_g) == pytest.approx(1.0, abs=1e-3)  # ancilla disentangled
    target = psi_g[:d] / np.linalg.norm(psi_g[:d])

    circuit = CompiledCircuit(
        betas=betas, rotations=rotations, target_state=target,
        infidelity=0.0, boundary_leakage=0.0,
    )
    pulse = build_circuit_pulse(circuit)
    state = simulate_circuit(pulse, n_cavity=n_cav, backend="dynamiqs")
    _, _, fid = eval_circuit(state, target, d, 1, n_cav)
    assert fid > 0.99
